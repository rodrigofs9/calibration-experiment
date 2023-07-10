from source.dataset.dataset import Dataset as MLDataset
from source.metrics.metrics_calculation import calculate_tradeoff_metrics
from source.popularity.popularity import Popularity
from surprise.reader import Reader
from sklearn.model_selection import train_test_split
from surprise import Trainset
from surprise import Dataset
from source.rerank.mmr import re_rank_list
from source.rerank.rerank_algo import run_user2, run_user3
from source.metrics.metrics import Metrics
import pandas as pd
import os
import argparse
import time
from tqdm import tqdm
import multiprocessing
import json
import gc
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.slope_one import SlopeOne
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.prediction_algorithms.matrix_factorization import SVDpp
from surprise.prediction_algorithms.matrix_factorization import NMF


import ast

from multiprocessing import Pool

from functools import partial

import traceback
import numpy as np


# with open("./data/ml-1m/ratios.json") as f:
#     ratios_division = json.loads(f.read())

indiceee = '1'

parser = argparse.ArgumentParser(description='Run the experiment.')

parser.add_argument(
    '--name',
    type=str,
    help='Name of the experiment',
    default=str(time.time())
)

args = parser.parse_args()

dataset = MLDataset()
# dataset.load_local_movielens_dataset("./ml-20m", type="ml20m_splitted", index=indiceee)
dataset.load_local_movielens_dataset("./yahoo", type='yahoo')

print("Dados Train")
print(dataset.train.shape)
print(len(dataset.train['item'].unique().tolist()))
print(len(dataset.train['user'].unique().tolist()))


print("Dados Test")
print(dataset.test.shape)
print(len(dataset.test['item'].unique().tolist()))
print(len(dataset.test['user'].unique().tolist()))

dataset = Popularity.generate_popularity_groups(dataset, subdivisions="mean", division='pareto')

### BUDGET
df_budget_revenue = pd.read_csv("./budget_yahoo2.csv")
df_budget_revenue = df_budget_revenue[df_budget_revenue['budget'] > 1]


movie_budget = {}
movie_revenue = {}

for index, row in df_budget_revenue.iterrows():
    movie_budget[row['item']] = row['budget']
    movie_revenue[row['item']] = row['revenue']

## POPULARITY BAGS
bags = 5
aux = np.log(df_budget_revenue['budget']).values
interval = max(aux)

popularity_bins = {}

values_budget = sorted(np.log(df_budget_revenue['budget']).values, reverse=False)
n_budget = len(values_budget)
for i in range(bags):
    popularity_bins[i] = (min(values_budget[i*n_budget//bags:(i+1)*n_budget//bags]), max(values_budget[i*n_budget//bags:(i+1)*n_budget//bags]))

print(popularity_bins)
### MOOVIE BINS

movies_bins = {}
movies_bins_profile = {}

def find_b(budget, popularity_bins):
    for bin_, (min_, max_) in popularity_bins.items():
        if budget >= min_ and budget <= max_:
            return bin_

for index, row in df_budget_revenue.iterrows():

    b = find_b(np.log(row['budget']), popularity_bins)

    movies_bins[int(row['item'])] = b
    movies_bins_profile[row['item']] = {b: 1}



### USER BINS

user_profile_bins = {}
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

for user in dataset.train['user'].unique():
    interacted = dataset.train[dataset.train['user'] == user]
    
    ppt = calculate_PPT(interacted['item'], popularity_bins, movie_budget, movies_bins)
    user_profile_bins[user] = ppt





def calc_user_ratio(dataset, user_id):

    interacted_by_user = dataset.train[dataset.train['user'] == user_id]['item']
    popularities = dataset.items[dataset.items['item'].isin(interacted_by_user.values)]
    high_pop = popularities[popularities['popularity'] == "H"]



    return str(user_id), high_pop.count()['popularity']/len(list(interacted_by_user))


def calc_user_profile(dataset, user_id):

    interacted_by_user = dataset.train[dataset.train['user'] == user_id].sort_values(by='item', ascending=False)
    prof = interacted_by_user[interacted_by_user['rating'] > 3]['item'].tolist()[:10]




    return user_id, prof



def calc_user_ratio2(dataset, user_id):
    BB_group = []
    N_group = []
    D_group = []
    interacted_by_user = dataset.train[dataset.train['user'] == user_id]['item']
    popularities = dataset.items[dataset.items['item'].isin(interacted_by_user.values)]
    high_pop = popularities[popularities['popularity'] == "H"]
    low_pop = popularities[popularities['popularity'] == "T"]

    if high_pop.count()['popularity']/len(list(interacted_by_user)) > 0.5:
        BB_group.append(user_id)
    elif low_pop.count()['popularity']/len(list(interacted_by_user)) > 0.5:
        N_group.append(user_id)
    else:
        D_group.append(user_id)


    return BB_group, N_group, D_group


qnt_users = len(dataset.train['user'].unique().tolist())
popularity_items = {}
popularity_all = {}
for item in dataset.train['item'].unique().tolist():
    popularity_items[item] = len(dataset.train[dataset.train['item'] == item])/qnt_users
    popularity_all[item] = np.log2(len(dataset.train[dataset.train['item'] == item]))

f = partial(calc_user_ratio, dataset)
pool = Pool(multiprocessing.cpu_count()-3)
aux = pool.map(
    f, dataset.train['user'].unique()
)
pool.close()
pool.join()

f = partial(calc_user_ratio2, dataset)
pool = Pool(multiprocessing.cpu_count()-3)
aux2 = pool.map(
    f, dataset.test['user'].unique()
)
pool.close()
pool.join()

ratios_division = {user : v for user, v in aux}
ratio_mean= sum(v for user, v in aux)

ratio_mean = ratio_mean/len(dataset.train['user'].unique())


f = partial(calc_user_profile, dataset)
pool = Pool(multiprocessing.cpu_count()-3)
aux3 = pool.map(
    f, dataset.test['user'].unique()
)
pool.close()
pool.join()

profiles = {user : v for user, v in aux3}




BB_group = []
N_group = []
D_group = []

for BB, N, D in aux2:
    if len(BB) > 0:
        BB_group.append(BB[0])
    if len(N) > 0:
        N_group.append(N[0])
    if len(D) > 0:
        D_group.append(D[0])

print("LEN BB", len(BB_group))
print("LEN N", len(N_group))
print("LEN D", len(D_group))


df_movies_meta = pd.read_csv("./yahoo/movie_db_yoda", sep="\t", encoding="ISO-8859-1", header=None).loc[:, [0, 1, 2]]
df_movies_meta.columns = ['item', 'title', 'description']

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm


df_movies_meta['item'] = df_movies_meta['item'].astype(int)
dataset.train['item'] = dataset.train['item'].astype(int)
dataset.test['item'] = dataset.test['item'].astype(int)

metadata = dict(zip(df_movies_meta['item'], df_movies_meta['description']))

corpus = [v for k,v in metadata.items()]
corpus = corpus[:len(corpus)//16]
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)
metadata = dict(zip(df_movies_meta['item'], vectorizer.transform(df_movies_meta['description'])))

occurrences = {}
n_items = dataset.train['item'].count()






for item in dataset.train['item'].unique():
    ratings = dataset.train[dataset.train['item'] == item]['item']
    occurrences[item] = ratings.count()/n_items


from source.rerank.mmr import mean_sd, ILS


def calculate_profile_scores(profiles, occurrences, vectorizer, user):
    user_profile_list = profiles[user]
    valmean = mean_sd(user_profile_list, occurrences)
    valils = ILS(user_profile_list, metadata, vectorizer)

    return user, valmean, valils


f = partial(calculate_profile_scores, profiles, occurrences, vectorizer)
pool = Pool(multiprocessing.cpu_count()-3)
aux4 = pool.map(
    f, dataset.test['user'].unique()
)
pool.close()
pool.join()

valils = {user : v2 for user, v, v2 in aux4}
valmean = {user : v for user, v, v2 in aux4}


def run_experiment(model_name_list, model_list, dataset, df, calibration_column_list=["genres"]):

    
    metrics_data = []

    all_genres = set()

    for i in dataset.items['genres']:
        for genre in i.split("|"):
            all_genres.add(genre)


    for ind in [int(indiceee)]:
        import time
        train = dataset.train
        test = dataset.test
        reader = Reader()

        trainset = Dataset.load_from_df(
            train[["user", "item", "rating"]], reader=reader
        ).build_full_trainset()
        
        


        print("Calculating items distribution")

        started = time.time()
        f = partial(Metrics.get_p_t_i_distribution_mp, dataset.items, "popularity")
        pool = Pool(multiprocessing.cpu_count()-3)
        p_t_i_all_items = pool.map(
            f, set(train["item"])
        )
        pool.close()
        pool.join()

        p_t_i_all_items = {id_:val for id_, val in p_t_i_all_items}

        print(f"Pop Finish: {time.time() - started}")

        pool = Pool(multiprocessing.cpu_count()-3)
        f = partial(Metrics.get_p_t_i_distribution_mp, dataset.items, "genres")
        p_t_i_genre_all_items = pool.map(
            f, set(train["item"])
        )
        pool.close()
        pool.join()
        p_t_i_genre_all_items = {id_:val for id_, val in p_t_i_genre_all_items}
        print(f"Ending items distribution calculation. Elapsed Time: {time.time() - started}")



        print("Calculating users distribution")
        start=time.time()
        ptu_pool = Pool(multiprocessing.cpu_count()-3)
        f = partial(Metrics.get_p_t_u_distribution_mp, train, dataset.items, "popularity", "rating", p_t_i_all_items)
        p_g_u_all_users = ptu_pool.map(
            f, set(test["user"])
        )
        ptu_pool.close()
        ptu_pool.join()
        p_g_u_all_users = {id_:val for id_, val in p_g_u_all_users}
        print(f"Pop Finish: {time.time() - start}")

        start=time.time()
        ptu_pool = Pool(multiprocessing.cpu_count()-3)
        f = partial(Metrics.get_p_t_u_distribution_mp, train, dataset.items, "genres", "rating", p_t_i_genre_all_items)
        p_g_u_genre_all_users = ptu_pool.map(
            f, set(test["user"])
        )
        ptu_pool.close()
        ptu_pool.join()
        p_g_u_genre_all_users = {id_:val for id_, val in p_g_u_genre_all_users}
        print(f"Genre Finish: {time.time() - start}")


        print("Calculating Users Custom Tradeoff")
        tradeoff_pool = Pool(multiprocessing.cpu_count()-3)
        started = time.time()
        cg_queue = [(user_id_, Metrics.tradeoff_genre_count(["H", "M", "T"],train,dataset.items,user_id_,distribution_column="popularity",p_g_u_all_users=p_g_u_all_users,)) for user_id_ in set(test["user"])]
        var_queue = [(user_id_, Metrics.tradeoff_variance(["H", "M", "T"],train,dataset.items,user_id_,distribution_column="popularity",p_g_u_all_users=p_g_u_all_users,)) for user_id_ in set(test["user"])]

        cg_genre_queue = [(user_id_, Metrics.tradeoff_genre_count(all_genres,train,dataset.items,user_id_,distribution_column="genres",p_g_u_all_users=p_g_u_genre_all_users,)) for user_id_ in set(test["user"])]
        var_genre_queue = [(user_id_, Metrics.tradeoff_variance(all_genres,train,dataset.items,user_id_,distribution_column="genres",p_g_u_all_users=p_g_u_genre_all_users,)) for user_id_ in set(test["user"])]
        print(f"[P1] Terminei de montar as listas {time.time() - started}")
        tradeoff_pool.close()
        tradeoff_pool.join()
        cg_all_users = {user_id_: result_async for user_id_, result_async in tqdm(cg_queue)}
        var_all_users = {user_id_: result_async for user_id_, result_async in tqdm(var_queue)}
        cg_genre_all_users = {user_id_: result_async for user_id_, result_async in tqdm(cg_genre_queue)}
        var_genre_all_users = {user_id_: result_async for user_id_, result_async in tqdm(var_genre_queue)}
        print(f"Ending users custom tradeoffs. Elapsed Time: {time.time() - started}")
        
        testratings = test
        trainratings = train

        recomended_list = {}

        import time

        
        for model_name, model in zip(model_name_list, model_list):
            model.fit(trainset)

            

            
            print("MODEL FITTED, STARTING EXPERIMENT")
            started = time.time()
            exp = Pool(2)
            f = partial(
                run_user3, train, dataset,
                calibration_column_list,
                model,
                trainratings,
                metadata,
                occurrences,
                vectorizer,
                profiles,
                valils, valmean

            )
            exp_results = exp.map(
                f, set(test["user"])
            )
            exp.close()
            exp.join()
            print(f"Ending experiment. Elapsed Time: {time.time() - started}")


            
            for calibration_column in calibration_column_list:
                single, comparison_type = calibration_column
                met = Pool(7)
                f = partial(
                    calculate_tradeoff_metrics,
                    model_name, ind,
                    f"{single}_{comparison_type}",
                    p_g_u_genre_all_users,
                    p_t_i_genre_all_items,
                    dataset,
                    trainratings,
                    p_g_u_all_users,
                    p_t_i_all_items,
                    test,
                    exp_results,
                    BB_group, N_group, D_group,
                    popularity_items,
                    popularity_bins,
                    movie_budget,
                    movies_bins,
                    user_profile_bins,
                movies_bins_profile,
                )
                tradeoffss = [0.5]
                if single is False:
                    tradeoffss = [0.25, 0.5, 0.75]

                met_results = met.map(
                    f, tradeoffss
                )
                met.close()
                met.join()
                for metrics_df in met_results:
                    metrics_data.append(metrics_df)
            
            del model
            gc.collect()
            pd.DataFrame(metrics_data).to_csv(f"./results/yahoo/jannach_tmp_{model_name}.csv")
    
    df = pd.concat([df, pd.DataFrame(metrics_data)])
    return df

df = pd.DataFrame(
    []
    )

models = []
models_names = []

##################### User knn
sim_options = {"name": "pearson_baseline", "user_based": True}
userknn = KNNWithMeans(k=30, sim_options=sim_options)
models.append(userknn)
models_names.append("userknn")

sim_options = {"name": "pearson_baseline", "user_based": False}
itemknn = KNNWithMeans(k=30, sim_options=sim_options)
models.append(itemknn)
models_names.append("itemknn")

# SO = SlopeOne()
# models.append(SO)
# models_names.append("so")

# svd = SVD(n_epochs=20, n_factors=20, lr_all=0.05, reg_all=0.02)
# models.append(svd)
# models_names.append("SVD")

# svdpp = SVDpp(n_epochs=20, n_factors=20, lr_all=0.005, reg_all=0.02)
# models.append(svdpp)
# models_names.append("SVDpp")

# nmf = NMF(n_epochs=50, n_factors=15, reg_bu=0.06, reg_bi=0.06)
# models.append(nmf)
# models_names.append("NMF")


df = run_experiment(models_names, models, dataset, df, calibration_column_list=[
    (True, 'ILS'),
    (True, 'mean'),
    (True, "aux"),
    (False, "ILS"),
    (False, 'aux')
])


df.columns=[
        'Model', "Fold", "Calibration Column",
        "Tradeoff", "MACE Genres", "MACE Pop",
        "LTC",  "MRMC Genres", "MRMC Pop", "MRMC Pop Budget",
        "AGGDIV", "Prec@10", "MAP@10", "MRR",
        "GAPBB", "GAPN", "GAPD"
        ]
df.reset_index()

df.to_csv(f"./results/yahoo/jannach_{time.time()}_complete_index1.csv")
