# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import math
import logging
DEFAULT_USER_COL = "user"
DEFAULT_ITEM_COL = "item"
DEFAULT_TIMESTAMP_COL = "timestamp"

logger = logging.getLogger(__name__)

try:
    from pyspark.sql import functions as F, Window
except ImportError:
    pass  # so the environment without spark doesn't break


def process_split_ratio(ratio):
    """Generate split ratio lists.

    Args:
        ratio (float or list): a float number that indicates split ratio or a list of float
        numbers that indicate split ratios (if it is a multi-split).

    Returns:
        tuple:
        - bool: A boolean variable multi that indicates if the splitting is multi or single.
        - list: A list of normalized split ratios.
    """
    if isinstance(ratio, float):
        if ratio <= 0 or ratio >= 1:
            raise ValueError("Split ratio has to be between 0 and 1")

        multi = False
    elif isinstance(ratio, list):
        if any([x <= 0 for x in ratio]):
            raise ValueError(
                "All split ratios in the ratio list should be larger than 0."
            )

        # normalize split ratios if they are not summed to 1
        if math.fsum(ratio) != 1.0:
            ratio = [x / math.fsum(ratio) for x in ratio]

        multi = True
    else:
        raise TypeError("Split ratio should be either float or a list of floats.")

    return multi, ratio


def min_rating_filter_pandas(
    data,
    min_rating=1,
    filter_by="user",
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
):
    """Filter rating DataFrame for each user with minimum rating.

    Filter rating data frame with minimum number of ratings for user/item is usually useful to
    generate a new data frame with warm user/item. The warmth is defined by min_rating argument. For
    example, a user is called warm if he has rated at least 4 items.

    Args:
        data (pandas.DataFrame): DataFrame of user-item tuples. Columns of user and item
            should be present in the DataFrame while other columns like rating,
            timestamp, etc. can be optional.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to
            filter with min_rating.
        col_user (str): column name of user ID.
        col_item (str): column name of item ID.

    Returns:
        pandas.DataFrame: DataFrame with at least columns of user and item that has been filtered by the given specifications.
    """
    split_by_column = _get_column_name(filter_by, col_user, col_item)

    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    return data.groupby(split_by_column).filter(lambda x: len(x) >= min_rating)


def min_rating_filter_spark(
    data,
    min_rating=1,
    filter_by="user",
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
):
    """Filter rating DataFrame for each user with minimum rating.

    Filter rating data frame with minimum number of ratings for user/item is usually useful to
    generate a new data frame with warm user/item. The warmth is defined by min_rating argument. For
    example, a user is called warm if he has rated at least 4 items.

    Args:
        data (pyspark.sql.DataFrame): DataFrame of user-item tuples. Columns of user and item
            should be present in the DataFrame while other columns like rating,
            timestamp, etc. can be optional.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to
            filter with min_rating.
        col_user (str): column name of user ID.
        col_item (str): column name of item ID.

    Returns:
        pyspark.sql.DataFrame: DataFrame with at least columns of user and item that has been filtered by the given specifications.
    """

    split_by_column = _get_column_name(filter_by, col_user, col_item)

    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    if min_rating > 1:
        window = Window.partitionBy(split_by_column)
        data = (
            data.withColumn("_count", F.count(split_by_column).over(window))
            .where(F.col("_count") >= min_rating)
            .drop("_count")
        )

    return data


def _get_column_name(name, col_user, col_item):
    if name == "user":
        return col_user
    elif name == "item":
        return col_item
    else:
        raise ValueError("name should be either 'user' or 'item'.")


def split_pandas_data_with_ratios(data, ratios, seed=42, shuffle=False):
    """Helper function to split pandas DataFrame with given ratios

    .. note::

        Implementation referenced from `this source <https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test>`_.

    Args:
        data (pandas.DataFrame): Pandas data frame to be split.
        ratios (list of floats): list of ratios for split. The ratios have to sum to 1.
        seed (int): random seed.
        shuffle (bool): whether data will be shuffled when being split.

    Returns:
        list: List of pd.DataFrame split by the given specifications.
    """
    if math.fsum(ratios) != 1.0:
        raise ValueError("The ratios have to sum to 1")

    split_index = np.cumsum(ratios).tolist()[:-1]

    if shuffle:
        data = data.sample(frac=1, random_state=seed)

    splits = np.split(data, [round(x * len(data)) for x in split_index])

    # Add split index (this makes splitting by group more efficient).
    for i in range(len(ratios)):
        splits[i]["split_index"] = i

    return splits


def filter_k_core(data, core_num=0, col_user="userID", col_item="itemID"):
    """Filter rating dataframe for minimum number of users and items by
    repeatedly applying min_rating_filter until the condition is satisfied.

    """
    num_users, num_items = len(data[col_user].unique()), len(data[col_item].unique())
    logger.info("Original: %d users and %d items", num_users, num_items)
    df_inp = data.copy()

    if core_num > 0:
        while True:
            df_inp = min_rating_filter_pandas(
                df_inp, min_rating=core_num, filter_by="item"
            )
            df_inp = min_rating_filter_pandas(
                df_inp, min_rating=core_num, filter_by="user"
            )
            count_u = df_inp.groupby(col_user)[col_item].count()
            count_i = df_inp.groupby(col_item)[col_user].count()
            if (
                len(count_i[count_i < core_num]) == 0
                and len(count_u[count_u < core_num]) == 0
            ):
                break
    df_inp = df_inp.sort_values(by=[col_user])
    num_users = len(df_inp[col_user].unique())
    num_items = len(df_inp[col_item].unique())
    logger.info("Final: %d users and %d items", num_users, num_items)

    return df_inp

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sk_split

def python_random_split(data, ratio=0.75, seed=42):
    """Pandas random splitter.

    The splitter randomly splits the input data.

    Args:
        data (pandas.DataFrame): Pandas DataFrame to be split.
        ratio (float or list): Ratio for splitting data. If it is a single float number
            it splits data into two halves and the ratio argument indicates the ratio
            of training data set; if it is a list of float numbers, the splitter splits
            data into several portions corresponding to the split ratios. If a list is
            provided and the ratios are not summed to 1, they will be normalized.
        seed (int): Seed.

    Returns:
        list: Splits of the input data as pandas.DataFrame.
    """
    multi_split, ratio = process_split_ratio(ratio)

    if multi_split:
        splits = split_pandas_data_with_ratios(data, ratio, shuffle=True, seed=seed)
        splits_new = [x.drop("split_index", axis=1) for x in splits]

        return splits_new
    else:
        return sk_split(data, test_size=None, train_size=ratio, random_state=seed)


def _do_stratification(
    data,
    ratio=0.75,
    min_rating=1,
    filter_by="user",
    is_random=True,
    seed=42,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_timestamp=DEFAULT_TIMESTAMP_COL,
):
    # A few preliminary checks.
    if not (filter_by == "user" or filter_by == "item"):
        raise ValueError("filter_by should be either 'user' or 'item'.")

    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    if col_user not in data.columns:
        raise ValueError("Schema of data not valid. Missing User Col")

    if col_item not in data.columns:
        raise ValueError("Schema of data not valid. Missing Item Col")

    if not is_random:
        if col_timestamp not in data.columns:
            raise ValueError("Schema of data not valid. Missing Timestamp Col")

    multi_split, ratio = process_split_ratio(ratio)

    split_by_column = col_user if filter_by == "user" else col_item

    ratio = ratio if multi_split else [ratio, 1 - ratio]

    if min_rating > 1:
        data = min_rating_filter_pandas(
            data,
            min_rating=min_rating,
            filter_by=filter_by,
            col_user=col_user,
            col_item=col_item,
        )

    if is_random:
        np.random.seed(seed)
        data["random"] = np.random.rand(data.shape[0])
        order_by = "random"
    else:
        order_by = col_timestamp

    data = data.sort_values([split_by_column, order_by])

    groups = data.groupby(split_by_column)

    data["count"] = groups[split_by_column].transform("count")
    data["rank"] = groups.cumcount() + 1

    if is_random:
        data = data.drop("random", axis=1)

    splits = []
    prev_threshold = None
    for threshold in np.cumsum(ratio):
        condition = data["rank"] <= round(threshold * data["count"])
        if prev_threshold is not None:
            condition &= data["rank"] > round(prev_threshold * data["count"])
        splits.append(data[condition].drop(["rank", "count"], axis=1))
        prev_threshold = threshold

    return splits


def python_chrono_split(
    data,
    ratio=0.75,
    min_rating=1,
    filter_by="user",
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_timestamp=DEFAULT_TIMESTAMP_COL,
):
    """Pandas chronological splitter.

    This function splits data in a chronological manner. That is, for each user / item, the
    split function takes proportions of ratings which is specified by the split ratio(s).
    The split is stratified.

    Args:
        data (pandas.DataFrame): Pandas DataFrame to be split.
        ratio (float or list): Ratio for splitting data. If it is a single float number
            it splits data into two halves and the ratio argument indicates the ratio of
            training data set; if it is a list of float numbers, the splitter splits
            data into several portions corresponding to the split ratios. If a list is
            provided and the ratios are not summed to 1, they will be normalized.
        seed (int): Seed.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to
            filter with min_rating.
        col_user (str): column name of user IDs.
        col_item (str): column name of item IDs.
        col_timestamp (str): column name of timestamps.

    Returns:
        list: Splits of the input data as pandas.DataFrame.
    """
    return _do_stratification(
        data,
        ratio=ratio,
        min_rating=min_rating,
        filter_by=filter_by,
        col_user=col_user,
        col_item=col_item,
        col_timestamp=col_timestamp,
        is_random=False,
    )


def python_stratified_split(
    data,
    ratio=0.75,
    min_rating=1,
    filter_by="user",
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    seed=42,
):
    """Pandas stratified splitter.

    For each user / item, the split function takes proportions of ratings which is
    specified by the split ratio(s). The split is stratified.

    Args:
        data (pandas.DataFrame): Pandas DataFrame to be split.
        ratio (float or list): Ratio for splitting data. If it is a single float number
            it splits data into two halves and the ratio argument indicates the ratio of
            training data set; if it is a list of float numbers, the splitter splits
            data into several portions corresponding to the split ratios. If a list is
            provided and the ratios are not summed to 1, they will be normalized.
        seed (int): Seed.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to
            filter with min_rating.
        col_user (str): column name of user IDs.
        col_item (str): column name of item IDs.

    Returns:
        list: Splits of the input data as pandas.DataFrame.
    """
    return _do_stratification(
        data,
        ratio=ratio,
        min_rating=min_rating,
        filter_by=filter_by,
        col_user=col_user,
        col_item=col_item,
        is_random=True,
        seed=seed,
    )


def numpy_stratified_split(X, ratio=0.75, seed=42):
    """Split the user/item affinity matrix (sparse matrix) into train and test set matrices while maintaining
    local (i.e. per user) ratios.

    Main points :

    1. In a typical recommender problem, different users rate a different number of items,
    and therefore the user/affinity matrix has a sparse structure with variable number
    of zeroes (unrated items) per row (user). Cutting a total amount of ratings will
    result in a non-homogeneous distribution between train and test set, i.e. some test
    users may have many ratings while other very little if none.

    2. In an unsupervised learning problem, no explicit answer is given. For this reason
    the split needs to be implemented in a different way then in supervised learningself.
    In the latter, one typically split the dataset by rows (by examples), ending up with
    the same number of features but different number of examples in the train/test setself.
    This scheme does not work in the unsupervised case, as part of the rated items needs to
    be used as a test set for fixed number of users.

    Solution:

    1. Instead of cutting a total percentage, for each user we cut a relative ratio of the rated
    items. For example, if user1 has rated 4 items and user2 10, cutting 25% will correspond to
    1 and 2.6 ratings in the test set, approximated as 1 and 3 according to the round() function.
    In this way, the 0.75 ratio is satisfied both locally and globally, preserving the original
    distribution of ratings across the train and test set.

    2. It is easy (and fast) to satisfy this requirements by creating the test via element subtraction
    from the original dataset X. We first create two copies of X; for each user we select a random
    sample of local size ratio (point 1) and erase the remaining ratings, obtaining in this way the
    train set matrix Xtst. The train set matrix is obtained in the opposite way.

    Args:
        X (numpy.ndarray, int): a sparse matrix to be split
        ratio (float): fraction of the entire dataset to constitute the train set
        seed (int): random seed

    Returns:
        numpy.ndarray, numpy.ndarray:
        - Xtr: The train set user/item affinity matrix.
        - Xtst: The test set user/item affinity matrix.
    """

    np.random.seed(seed)  # set the random seed
    test_cut = int((1 - ratio) * 100)  # percentage of ratings to go in the test set

    # initialize train and test set matrices
    Xtr = X.copy()
    Xtst = X.copy()

    # find the number of rated movies per user
    rated = np.sum(Xtr != 0, axis=1)

    # for each user, cut down a test_size% for the test set
    tst = np.around((rated * test_cut) / 100).astype(int)

    for u in range(X.shape[0]):
        # For each user obtain the index of rated movies
        idx = np.asarray(np.where(Xtr[u] != 0))[0].tolist()

        # extract a random subset of size n from the set of rated movies without repetition
        idx_tst = np.random.choice(idx, tst[u], replace=False)
        idx_train = list(set(idx).difference(set(idx_tst)))

        # change the selected rated movies to unrated in the train set
        Xtr[u, idx_tst] = 0
        # set the movies that appear already in the train set as 0
        Xtst[u, idx_train] = 0

    del idx, idx_train, idx_tst

    return Xtr, Xtst