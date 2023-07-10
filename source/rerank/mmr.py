import numpy as np

from source.metrics.metrics import Metrics

list_recomended_items = [("item", "rating")]

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
from numba import jit
from scipy.spatial.distance import jensenshannon

def cos_sim(a, b):
    return cosine_similarity(a, b)[0][0]



@jit(nopython=True, cache=True)
def cosine_similarity_numba(u:np.ndarray, v:np.ndarray):
    assert(u.shape[0] == v.shape[0])
    uv: np.float64 = 0.0
    uu: np.float64 = 0.0
    vv: np.float64 = 0.0
    for i in range(u.shape[0]):
        uv += u[i] @ v[i]
        uu += u[i] @ u[i]
        vv += v[i] @ v[i]
    cos_theta: np.float64 = 1.0
    if uu!=0 and vv!=0:
        cos_theta = uv/np.sqrt(uu*vv)
    return cos_theta




def ILS(list_items, metadata, vectorizer):
    sum_ = 0
    n = len(list_items)
    n_2 = n**2
    for i in range(n):
        for j in range(i, n):
            sum_ += 2*cosine_similarity_numba(metadata[list_items[j]].toarray().astype(np.float64), metadata[list_items[i]].toarray().astype(np.float64),)/n_2

    # sum_ = sum([sum([cosine_similarity_numba(metadata[j].toarray().astype(np.float64), metadata[i].toarray().astype(np.float64),)/n_2 for i in list_items]) for j in list_items])
    return sum_ 


def ILS_increment(list_items, new_item, calculate_ils, metadata, vectorizer):
    sum_ = 0
    n = len(list_items)+1
    n_2 = n**2
    for i in list_items:
        calculate_ils += 2*cosine_similarity_numba(metadata[new_item].toarray().astype(np.float64), metadata[i].toarray().astype(np.float64),)/n_2
    return calculate_ils + 1/n_2


def mean_sd(list_items, ocurrences):
    
    ocur = [ocurrences[i] for i in list_items]
    mean_ = np.mean(ocur)
    sd_ = np.std(ocur)
    
    return mean_ + sd_

def EMD(recommended_list, profile_list, ocurrences):
    ocur_r = [ocurrences[i] for i in recommended_list]
    ocur_p = [ocurrences[i] for i in profile_list]

    if len(ocur_r) == 0 or len(ocur_p) == 0:
        return 1
    return wasserstein_distance(ocur_r, ocur_p)

import time

def error(recommended_list, profile_list, metadata, ocurrentes, comparison_type='ILS', single=True, tradeoff = 0.5, vectorizer=None, valils=None, valmean=None, calculate_ils=0):
    
    
    if single is True:
        if comparison_type == 'ILS':
            # val1 = ILS(recommended_list, metadata, vectorizer)
            val1 = ILS_increment(recommended_list[:-1], recommended_list[-1], calculate_ils, metadata, vectorizer)
            val2 = valils

            return abs(val1-val2), val1
        elif comparison_type == 'mean':
            val1 = mean_sd(recommended_list, ocurrentes)
            val2 = valmean
            return abs(val1-val2), None
        else:
            emd =  EMD(recommended_list, profile_list, ocurrentes)
            return emd, None
    else:
        if comparison_type not in ['ILS', 'mean']:
            # val1_1 = ILS(recommended_list, metadata, vectorizer)
            val1_1 = ILS_increment(recommended_list[:-1], recommended_list[-1], calculate_ils, metadata, vectorizer)
            val2_1 = valils
            val1_2 = mean_sd(recommended_list, ocurrentes)
            val2_2 = valmean
            return tradeoff* abs(val1_1-val2_1) - (1-tradeoff)*abs(val1_2-val2_2), val1_1
        else:
            # val1_1 = ILS(recommended_list, metadata, vectorizer)
            val1_1 = ILS_increment(recommended_list[:-1], recommended_list[-1], calculate_ils, metadata, vectorizer)
            val2_1 = valils
            
            val = EMD(recommended_list, profile_list, ocurrentes)
            return tradeoff* abs(val1_1-val2_1) - (1-tradeoff)*val, val1_1


def calculate_calibration_sum(
    movies_data,
    dataset,
    temporary_list,
    p_g_u,
    alpha=0.01,
    distribution_column="genres",
    p_t_i_all_items=None,
    calibration_type="kl"
):
    interacted_distr = p_g_u
    reco_distr = Metrics.get_q_t_u_distribution(
        [(id_, index + 1) for index, id_ in enumerate(temporary_list)],
        movies_data,
        distribution_column=distribution_column,
        p_t_i_all_items=p_t_i_all_items,
    )
    if calibration_type == 'jansen':
        vector = [(interacted_distr.get(pop, 0.0), reco_distr.get(pop, 0.0)) for pop in ['H', 'M', 'T']]
        p_vector = [i[0] for i in vector]
        q_vector = [i[1] for i in vector]
        try:
            vv = jensenshannon(p_vector, q_vector)
            return vv
        except:
            print("erro jensen")
    else:

        kl_div = 0.0
        for genre, p in interacted_distr.items():
            q = reco_distr.get(genre, 0.0)
            til_q = (1 - alpha) * q + alpha * p

            if p == 0.0 or til_q == 0.0:
                kl_div = kl_div
            else:
                kl_div = kl_div + (p * np.log2(p / til_q))
        return kl_div


import time


def re_rank_list(
    trainratings,
    movies_data,
    user_id,
    list_recomended_items,
    tradeoff=0.5,
    N=10,
    p_g_u_all_users=None,
    p_t_i_all_items=None,
    distribution_column="popularity",
    weight="rating",
    calibration_type="kl"
):
    re_ranked_list = []
    re_ranked_with_score = []

    # list_recomended_items = sorted(list_recomended_items, key=lambda x: x[1], reverse=False)
    if p_g_u_all_users is None:
        p_g_u = Metrics.get_p_t_u_distribution(
            trainratings,
            movies_data,
            user_id,
            distribution_column=distribution_column,
            weight=weight,
        )
    else:
        p_g_u = p_g_u_all_users[user_id]

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
                trainratings,
                temporary_list,
                p_g_u,
                distribution_column=distribution_column,
                p_t_i_all_items=p_t_i_all_items,
                calibration_type=calibration_type
            )
            maximized = (1 - tradeoff) * weight_part - tradeoff * (
                full_tmp_calib
            )
            if maximized > max_mmr:
                max_mmr = maximized
                max_item = item
                max_item_rating = rating
            
        if max_item is not None:
            re_ranked_list.append(max_item)
            re_ranked_with_score.append((max_item, max_item_rating))

    return re_ranked_list, re_ranked_with_score

import time

def re_rank_list2(
    trainratings,
    movies_data,
    user_id,
    list_recomended_items,
    N=10,
    user_profile_list=None,
    metadata=None,
    occurrences=None,
    comparison_type=None,
    tradeoff=0.5,
    single=True,
    vectorizer=None, valils=None, valmean=None
):

    re_ranked_list = []
    re_ranked_with_score = []

    valils = valils[user_id]
    valmean = valmean[user_id]

    val1 = 0

    for index_N in range(N):
        max_mmr = np.inf
        max_item = None
        max_item_rating = None
        max_val = None
        for item, rating in list_recomended_items:
            if item in re_ranked_list:
                continue

            temporary_list = re_ranked_list + [item]
            
            maximized, val1 = error(
                temporary_list, user_profile_list,
                metadata=metadata,
                ocurrentes=occurrences,
                comparison_type=comparison_type,
                single=single,
                tradeoff=tradeoff,
                vectorizer=vectorizer,
                valmean=valmean,
                valils=valils,
                calculate_ils=val1

            )
            


            if maximized <= max_mmr:
                max_mmr = maximized
                max_item = item
                max_item_rating = rating
                max_val = val1

        if max_item is not None:
            re_ranked_list.append(max_item)
            re_ranked_with_score.append((max_item, max_item_rating))
            val1 = max_val

        else:
            re_ranked_list.append(list_recomended_items[index_N][0])
            re_ranked_with_score.append((list_recomended_items[index_N][0], list_recomended_items[index_N][1]))




    return re_ranked_list, re_ranked_with_score

def calculate_PPT(list_, popularity_bins, movie_budget, movies_bins):
    ppt = {k:0 for k in popularity_bins.keys()}
    sum_ = 0
    for item in list_:
        
        sum_ += np.log(movie_budget.get(item, 1))
        b = movies_bins.get(item, 0)
        ppt[b] += np.log(movie_budget.get(item, 1))
    
    
    for i in ppt.keys():
        ppt[i] = ppt[i]/sum_ if sum_ > 0 else 0
        
    return ppt

def calculate_calibration_sum_PPTM(
    movies_data,
    dataset,
    temporary_list,
    p_g_u,
    alpha=0.01,
    popularity_bins=None,
    movie_budget=None,
    movies_bins=None
):

    interacted_distr = p_g_u

    reco_distr = calculate_PPT(temporary_list, popularity_bins, movie_budget, movies_bins)

    kl_div = 0.0
    for genre, p in interacted_distr.items():
        q = reco_distr.get(genre, 0.0)
        til_q = (1 - alpha) * q + alpha * p

        if p == 0.0 or til_q == 0.0:
            kl_div = kl_div
        else:
            kl_div = kl_div + (p * np.log2(p / til_q))
    return kl_div

def re_rank_list_PPTM(
    trainratings,
    movies_data,
    user_id,
    list_recomended_items,
    tradeoff=0.5,
    N=10,
    p_g_u_all_users=None,
    p_t_i_all_items=None,
    distribution_column="popularity",
    weight="rating",
    popularity_bins=None,
    movie_budget=None,
    movies_bins=None
):

    re_ranked_list = []
    re_ranked_with_score = []

    

    # list_recomended_items = sorted(list_recomended_items, key=lambda x: x[1], reverse=False)

    p_g_u = p_g_u_all_users.get(user_id, {})

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
            full_tmp_calib = calculate_calibration_sum_PPTM(
                movies_data,
                trainratings,
                temporary_list,
                p_g_u,
                popularity_bins=popularity_bins,
                movie_budget=movie_budget,
                movies_bins=movies_bins
            )
            maximized = (1 - tradeoff) * weight_part - tradeoff * (
                full_tmp_calib
            )
            if maximized > max_mmr:
                max_mmr = maximized
                max_item = item
                max_item_rating = rating

        if max_item is not None:
            re_ranked_list.append(max_item)
            re_ranked_with_score.append((max_item, max_item_rating))

    return re_ranked_list, re_ranked_with_score


import time