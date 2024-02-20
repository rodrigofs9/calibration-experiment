import pandas as pd
from surprise.reader import Reader
from surprise import Dataset
from calibration import re_rank_list
from baselines.pairwise.pairwise import PairWise

reader = Reader()

def run_in_processing_rerank(train, test, dataset, users, items, user, tradeoff = 0.0):
    recommend_results = {
        'inprocessing': {
            tradeoff: {
                'reranked': {}
            }
        }
    }
    items_metadata = dataset
    category_per_item = {}
    category_map = {
        'M': 1,
        'H': 0,
        'T': 1
    }
    for _, row in items_metadata.iterrows():
        category_per_item[row['item']] = category_map[row['popularity']]

    users__ = []

    model = PairWise(
        users, items,
        train, test,
        category_per_item,
        'item',
        'user',
        'rating'
    )
    model.train()
    model.predict()

    predictions = model.test()

    for user_id_, y_pred in predictions.items():
        users__.append(user_id_)
        result = y_pred.argsort()[-10:][::-1]
        result = result[0][:10]
        result_tuples = [(valor, indice + 1) for indice, valor in enumerate(result)]

        recommend_results['inprocessing'][0.0]['reranked'][user] = result_tuples

    return user, recommend_results

def run_rerank(
    train, dataset,
    calibration_type,
    model,
    trainratings,
    target_all_users_genres_distribution,
    target_all_users_distribution,
    target_all_items_genres_distribution,
    target_all_items_distribution,
    var_genre_all_users,
    cg_genre_all_users,
    var_all_users,
    cg_all_users,
    ratios_division,
    ratio_mean,
    calibration_calculation_type,
    user
):
    try:
        recommend_results = {}
        know_items_ids = (train[train["user"] == user]["item"].unique().tolist())

        data = {"item": list(set(dataset.items["item"]) - set(know_items_ids))}

        user_testset_df = pd.DataFrame(data)
        user_testset_df["user"] = user
        user_testset_df["rating"] = 0.0

        testset = (
            Dataset.load_from_df(
                user_testset_df[["user", "item", "rating"]],
                reader=reader,
            )
            .build_full_trainset()
            .build_testset()
        )
        predictions = model.test(testset)
        sorted_recommendations = sorted(
                    [
                        (pred.iid, pred.est)
                        for pred in predictions
                        if pred.uid == user
                    ],
                    key = lambda x: x[1],
                    reverse = True,
        )

        recommended_user = sorted_recommendations

        if calibration_type not in recommend_results:
            recommend_results[calibration_type] = {}

        for tradeoff in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            if tradeoff not in recommend_results[calibration_type]:
                recommend_results[calibration_type][tradeoff] = {"reranked": {}}

            tradeoff_value = tradeoff

            re_ranked = None
            if calibration_type == "genre":

                if tradeoff == "VAR":
                    tradeoff_value = var_genre_all_users[user]
                elif tradeoff == "GC":
                    tradeoff_value = cg_genre_all_users[user]
                
                re_ranked, _ = re_rank_list(
                trainratings,
                dataset.items,
                user,
                recommended_user[:100],
                target_all_users_distribution = target_all_users_genres_distribution,
                target_all_items_distribution = target_all_items_genres_distribution,
                tradeoff = tradeoff_value,
                N = 10,
                distribution_column = "genres",
                calibration_type = calibration_type
            )
            elif calibration_type == "double":
                if tradeoff == "VAR":
                    tradeoff_value = var_genre_all_users[user]
                elif tradeoff == "GC":
                    tradeoff_value = cg_genre_all_users[user]
                
                re_ranked, first_calibration  = re_rank_list(
                trainratings,
                dataset.items,
                user,
                recommended_user[:100],
                target_all_users_distribution = target_all_users_genres_distribution,
                target_all_items_distribution = target_all_items_genres_distribution,
                tradeoff = tradeoff_value,
                N = 100,
                distribution_column = "popularity",
                calibration_type = calibration_type
                )
                
                re_ranked, _  = re_rank_list(
                trainratings,
                dataset.items,
                user,
                first_calibration,
                target_all_users_distribution = target_all_users_genres_distribution,
                target_all_items_distribution = target_all_items_genres_distribution,
                tradeoff = tradeoff_value,
                N = 10,
                distribution_column = "genres",
                calibration_type = calibration_type
                )
            elif calibration_type == 'personalized':
                if ratios_division[str(user)] >= ratio_mean :
                    if tradeoff == "VAR":
                        tradeoff_value = var_genre_all_users[user]
                    elif tradeoff == "GC":
                        tradeoff_value = cg_genre_all_users[user]
                    
                    re_ranked, _ = re_rank_list(
                    trainratings,
                    dataset.items,
                    user,
                    recommended_user[:100],
                    target_all_users_distribution = target_all_users_genres_distribution,
                    target_all_items_distribution = target_all_items_genres_distribution,
                    tradeoff = tradeoff_value,
                    N = 10,
                    distribution_column = "genres",
                    calibration_type = calibration_type
                )
                else:
                    if tradeoff == "VAR":
                        tradeoff_value = var_all_users[user]
                    elif tradeoff == "GC":
                        tradeoff_value = cg_all_users[user]

                    re_ranked, _ = re_rank_list(
                    trainratings,
                    dataset.items,
                    user,
                    recommended_user[:100],
                    target_all_users_distribution = target_all_users_distribution,
                    target_all_items_distribution = target_all_items_distribution,
                    tradeoff = tradeoff_value,
                    N = 10,
                    distribution_column = "popularity",
                    calibration_type = calibration_type
                )
            elif calibration_type == 'inv_personalized':
                if ratios_division[str(user)] <= ratio_mean :
                    if tradeoff == "VAR":
                        tradeoff_value = var_genre_all_users[user]
                    elif tradeoff == "GC":
                        tradeoff_value = cg_genre_all_users[user]
                    
                    re_ranked, _ = re_rank_list(
                    trainratings,
                    dataset.items,
                    user,
                    recommended_user[:100],
                    target_all_users_distribution = target_all_users_genres_distribution,
                    target_all_items_distribution = target_all_items_genres_distribution,
                    tradeoff = tradeoff_value,
                    N = 10,
                    distribution_column = "genres",
                    calibration_type = calibration_type
                )
                else:
                    if tradeoff == "VAR":
                        tradeoff_value = var_all_users[user]
                    elif tradeoff == "GC":
                        tradeoff_value = cg_all_users[user]

                    re_ranked, _ = re_rank_list(
                    trainratings,
                    dataset.items,
                    user,
                    recommended_user[:100],
                    target_all_users_distribution = target_all_users_distribution,
                    target_all_items_distribution = target_all_items_distribution,
                    tradeoff = tradeoff_value,
                    N = 10,
                    distribution_column = "popularity",
                    calibration_type = calibration_type
                )
            else:

                if tradeoff == "VAR":
                    tradeoff_value = var_all_users[user]
                elif tradeoff == "GC":
                    tradeoff_value = cg_all_users[user]

                re_ranked, _ = re_rank_list(
                    trainratings,
                    dataset.items,
                    user,
                    recommended_user[:100],
                    target_all_users_distribution = target_all_users_distribution,
                    target_all_items_distribution = target_all_items_distribution,
                    tradeoff = tradeoff_value,
                    N = 10,
                    distribution_column = "popularity",
                    calibration_type = calibration_type
                )
            
            recommend_results[calibration_type][tradeoff]["reranked"][user] = [
                (item, index + 1)
                for index, item in enumerate(re_ranked)
            ]
        return user, recommend_results
    except Exception as e:
            print("An error occurred in run_user:")
            print(e)

def run_bpr_rerank(
    train, dataset,
    model,
    tradeoff,
    user
):
    try:
        recommend_results = {
            'top10': {
                tradeoff: {
                    'reranked': {}
                }
            }
        }

        know_items_ids = (
            train[train["user"] == user]["item"].unique().tolist()
        )

        data = {
            "item": list(
                set(dataset.items["item"]) - set(know_items_ids)
            )
        }

        user_testset_df = pd.DataFrame(data)
        user_testset_df["user"] = user
        user_testset_df["rating"] = 0.0

        testset = (
            Dataset.load_from_df(
                user_testset_df[["user", "item", "rating"]],
                reader=reader,
            )
            .build_full_trainset()
            .build_testset()
        )
        predictions = model.test(testset)
        aux_3 = sorted(
            [
                (pred[0], pred[1])
                for pred in predictions
            ],
            key = lambda x: x[1],
            reverse = True,
        )

        sorted_list = sorted(aux_3, key = lambda x: x[1], reverse = True)
        top_10 = [(item[0], i + 1) for i, item in enumerate(sorted_list[:10])]

        recommend_results['top10'][tradeoff]["reranked"][user] = [
                            item
                            for _, item in enumerate(top_10)
                        ]
        
        return user, recommend_results
    except Exception as e:
            print("An error occurred in run_user:")
            print(e)
