import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

from source.dataset.dataset import Dataset as MLDataset
from source.metrics.metrics_calculation_pairwise import calculate_tradeoff_metrics
from source.popularity.popularity import Popularity
from surprise.reader import Reader
from sklearn.model_selection import train_test_split
from surprise import Trainset
from surprise import Dataset
from source.rerank.mmr import re_rank_list
from source.rerank.rerank_algo import run_user_in_processing_mitigation
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
import random

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

def run_experiment(dataset, df):
    metrics_data = []
    all_genres = set()

    dataset.train['user' + '_original'] = dataset.train['user'] # Ensure that we save the original user ids
    dataset.train['item' + '_original'] = dataset.train['item'] # Ensure that we save the original item ids
    dataset.train['user'] = dataset.train['user'].astype('category').cat.codes # Ensure that user ids are in [0, |U|] 
    dataset.train['item'] = dataset.train['item'].astype('category').cat.codes # Ensure that item ids are in [0, |I|] 

    dataset.test['user' + '_original'] = dataset.test['user'] # Ensure that we save the original user ids
    dataset.test['item' + '_original'] = dataset.test['item'] # Ensure that we save the original item ids
    dataset.test['user'] = dataset.test['user'].astype('category').cat.codes # Ensure that user ids are in [0, |U|] 
    dataset.test['item'] = dataset.test['item'].astype('category').cat.codes # Ensure that item ids are in [0, |I|] 

    newDataset = dataset.items
    newDataset['item' + '_original'] = newDataset['item'] # Ensure that we save the original item ids
    #newDataset['item'] = newDataset['item'].astype('category').cat.codes # Ensure that item ids are in [0, |I|] 
    newDataset = newDataset.merge(dataset.train[['item', 'item_original']], on='item_original', how='left')
    newDataset['item'] = newDataset['item_y']

    for ind in [int(fold)]:
        import time
        train = dataset.train
        test = dataset.test
        reader = Reader()

        trainset = Dataset.load_from_df(
            train[["user", "item", "rating"]], reader=reader
        ).build_full_trainset()
        
        print("Calculating items distribution")

        started = time.time()
        metricsInstance = Metrics()
        f = partial(metricsInstance.get_p_t_i_distribution_mp, newDataset, "popularity")
        pool = Pool(multiprocessing.cpu_count()-3)
        p_t_i_all_items = pool.map(
            f, set(train["item"])
        )
        pool.close()
        pool.join()

        p_t_i_all_items = {id_:val for id_, val in p_t_i_all_items}

        print(f"Pop Finish: {time.time() - started}")

        pool = Pool(multiprocessing.cpu_count()-3)
        f = partial(metricsInstance.get_p_t_i_distribution_mp, newDataset, "genres")
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
        f = partial(metricsInstance.get_p_t_u_distribution_mp, train, newDataset, "popularity", "rating", p_t_i_all_items)
        p_g_u_all_users = ptu_pool.map(
            f, set(test["user"])
        )
        ptu_pool.close()
        ptu_pool.join()
        p_g_u_all_users = {id_:val for id_, val in p_g_u_all_users}
        print(f"Pop Finish: {time.time() - start}")

        start=time.time()
        ptu_pool = Pool(multiprocessing.cpu_count()-3)
        f = partial(metricsInstance.get_p_t_u_distribution_mp, train, newDataset, "genres", "rating", p_t_i_genre_all_items)
        p_g_u_genre_all_users = ptu_pool.map(
            f, set(test["user"])
        )
        ptu_pool.close()
        ptu_pool.join()
        p_g_u_genre_all_users = {id_:val for id_, val in p_g_u_genre_all_users}
        print(f"Genre Finish: {time.time() - start}")

        tradeoff_pool = Pool(multiprocessing.cpu_count()-3)
        started = time.time()
        
        testratings = test
        trainratings = train
        recomended_list = {}

        print("MODEL FITTED, STARTING EXPERIMENT")
        started = time.time()
        exp = Pool(4)

        #users = dataset.train['user'].unique().tolist()
        users = list(np.unique(dataset.train['user'].values))
        items = list(np.unique(dataset.train['item'].values))
        from itertools import islice

        f = partial(run_user_in_processing_mitigation, train, test, newDataset, users, items)
        #exp_results = exp.map(f, set(test["user"]))
        exp_results = exp.map(f, list(islice(test["user"], 100)))
        exp.close()
        exp.join()
        print(f"Ending experiment. Elapsed Time: {time.time() - started}")

        #print('popularity_items')
        #print(popularity_items)

        #datasetItems = pd.DataFrame(newDataset)

        met = Pool(7)
        f = partial(
            calculate_tradeoff_metrics,
            'pairwise', ind,
            'inprocessing',
            p_g_u_genre_all_users,
            p_t_i_genre_all_items,
            newDataset,
            trainratings,
            p_g_u_all_users,
            p_t_i_all_items,
            test,
            exp_results,
            BB_group, N_group, D_group,
            popularity_items,
            None,
            None,
            None,
            None,
            None,
            True,
        )

        met_results = met.map(
            f, [0.0]
        )
        met.close()
        met.join()
        for metrics_df in met_results:
            metrics_data.append(metrics_df)
        
        gc.collect()
        pd.DataFrame(metrics_data).to_csv(f"./results/yahoo_song/pairwise_tmp.csv")
    
    df = pd.concat([df, pd.DataFrame(metrics_data)])
    return df

if __name__ == '__main__':
    fold = '3'

    parser = argparse.ArgumentParser(description='Run the experiment.')

    parser.add_argument(
        '--name',
        type=str,
        help='Name of the experiment',
        default=str(time.time())
    )

    args = parser.parse_args()

    dataset = MLDataset()
    dataset.load_local_movielens_dataset("./datasets/yahoo_song", type='yahoo_song_new', index='7')

    print("Dados Train")
    print(dataset.train.shape)
    print(len(dataset.train['item'].unique().tolist()))
    print(len(dataset.train['user'].unique().tolist()))

    print("Dados Test")
    print(dataset.test.shape)
    print(len(dataset.test['item'].unique().tolist()))
    print(len(dataset.test['user'].unique().tolist()))

    dataset = Popularity.generate_popularity_groups(dataset, subdivisions="mean", division='pareto')

    if dataset is not None:
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
            f, dataset.train['user'].unique()
        )
        pool.close()
        pool.join()

        ratios_division = {user : v for user, v in aux}
        ratio_mean= sum(v for user, v in aux)

        ratio_mean = ratio_mean/len(dataset.train['user'].unique())

        f = partial(calc_user_profile, dataset)
        pool = Pool(multiprocessing.cpu_count()-3)
        aux3 = pool.map(
            f, dataset.train['user'].unique()
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

        df = pd.DataFrame([])

        df = run_experiment(dataset, df)

        df.columns=[
                'Model', "Fold", "Calibration Column",
                "Tradeoff", "MACE Genres", "MACE Pop",
                "LTC",  "MRMC Genres", "MRMC Pop",
                "AGGDIV", "Prec@10", "MAP@10", "MRR",
                "GAPBB", "GAPN", "GAPD"
                ]
        df.reset_index()

        df.to_csv(f"./results/yahoo_song/pairwise_{time.time()}_complete_index1.csv")