import numpy as np

from metrics import Metrics
from scipy.spatial.distance import jensenshannon

list_recomended_items = [("item", "rating")]

def calculate_calibration_sum(
    movies_data,
    temporary_list,
    target_user_distribution,
    alpha = 0.01,
    distribution_column = "genres",
    target_all_items_distribution = None,
    calibration_type = "kl"
):
    
    recommended_user_distribution = Metrics.calculate_recommended_user_distribution(
        [(id_, index + 1) for index, id_ in enumerate(temporary_list)],
        movies_data,
        distribution_column = distribution_column,
        target_all_items_distribution = target_all_items_distribution,
    )
    if calibration_type == "jansen":
        vector = [(target_user_distribution.get(pop, 0.0), recommended_user_distribution.get(pop, 0.0)) for pop in ['H', 'M', 'T']]
        p_vector = [i[0] for i in vector]
        q_vector = [i[1] for i in vector]
        try:
            vv = jensenshannon(p_vector, q_vector)
            return vv
        except:
            print("erro jensen")
    else:
        kl_div = 0.0
        for genre, p in target_user_distribution.items():
            q = recommended_user_distribution.get(genre, 0.0)
            til_q = (1 - alpha) * q + alpha * p

            if p == 0.0 or til_q == 0.0:
                kl_div = kl_div
            else:
                kl_div = kl_div + (p * np.log2(p / til_q))
        return kl_div

def re_rank_list(
    trainratings,
    movies_data,
    user_id,
    list_recomended_items,
    tradeoff = 0.5,
    N = 10,
    target_all_users_distribution = None,
    target_all_items_distribution = None,
    distribution_column = "popularity",
    weight = "rating",
    calibration_type = "kl"
):
    re_ranked_list = []
    re_ranked_with_score = []

    if target_all_users_distribution is None:
        target_user_distribution = Metrics.calculate_target_user_distribution(
            trainratings,
            movies_data,
            distribution_column = distribution_column,
            weight = weight,
            user_id = user_id
        )
    else:
        target_user_distribution = target_all_users_distribution[user_id]

    for _ in range(N):
        max_mmr = -np.inf
        max_item = None
        max_item_rating = None
        for item, rating in list_recomended_items:
            if item in re_ranked_list:
                continue

            weight_part = sum(
                recomendation[1]
                for recomendation in (re_ranked_with_score + [(item, rating)])
            )

            temporary_list = re_ranked_list + [item]
            full_tmp_calib = calculate_calibration_sum(
                movies_data,
                temporary_list,
                target_user_distribution,
                distribution_column = distribution_column,
                target_all_items_distribution = target_all_items_distribution,
                calibration_type = calibration_type
            )
            maximized = (1 - tradeoff) * weight_part - tradeoff * (full_tmp_calib)
            if maximized > max_mmr:
                max_mmr = maximized
                max_item = item
                max_item_rating = rating
            
        if max_item is not None:
            re_ranked_list.append(max_item)
            re_ranked_with_score.append((max_item, max_item_rating))

    return re_ranked_list, re_ranked_with_score