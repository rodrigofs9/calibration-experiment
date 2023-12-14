import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

from source.dataset.dataset import Dataset as MLDataset
from source.metrics.metrics_calculation import calculate_tradeoff_metrics
from source.popularity.popularity import Popularity
from surprise.reader import Reader
from sklearn.model_selection import train_test_split
from surprise import Trainset
from surprise import Dataset
from source.rerank.mmr import re_rank_list
from source.rerank.rerank_algo import run_user_bpr
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
from bpr_files.bpr_pop import BPR as BPR_ideias
from bpr_files.bpr_pop2 import BPR as BPR_ideia1
from bpr_files.bpr import BPR as BPR_classic

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

# with open("./data/ml-1m/ratios.json") as f:
#     ratios_division = json.loads(f.read())

def run_experiment(model_name_list, model_list, dataset, df, tradeoff, calibration_column_list=["genres"]):
    metrics_data = []
    all_genres = set()

    for i in dataset.items['genres']:
        for genre in i.split("|"):
            all_genres.add(genre)
    print(all_genres)

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
    f = partial(metricsInstance.get_p_t_i_distribution_mp, dataset.items, "popularity")
    pool = Pool(multiprocessing.cpu_count()-3)
    p_t_i_all_items = pool.map(
        f, set(train["item"])
    )
    pool.close()
    pool.join()

    p_t_i_all_items = {id_:val for id_, val in p_t_i_all_items}

    print(f"Pop Finish: {time.time() - started}")

    pool = Pool(multiprocessing.cpu_count()-3)
    f = partial(metricsInstance.get_p_t_i_distribution_mp, dataset.items, "genres")
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
    f = partial(metricsInstance.get_p_t_u_distribution_mp, train, dataset.items, "popularity", "rating", p_t_i_all_items)
    p_g_u_all_users = ptu_pool.map(
        f, set(test["user"])
    )
    ptu_pool.close()
    ptu_pool.join()
    p_g_u_all_users = {id_:val for id_, val in p_g_u_all_users}
    print(f"Pop Finish: {time.time() - start}")

    start=time.time()
    ptu_pool = Pool(multiprocessing.cpu_count()-3)
    f = partial(metricsInstance.get_p_t_u_distribution_mp, train, dataset.items, "genres", "rating", p_t_i_genre_all_items)
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
    cg_queue = [(user_id_, metricsInstance.tradeoff_genre_count(["H", "M", "T"],train,dataset.items,user_id_,distribution_column="popularity",p_g_u_all_users=p_g_u_all_users,)) for user_id_ in set(test["user"])]
    var_queue = [(user_id_, metricsInstance.tradeoff_variance(["H", "M", "T"],train,dataset.items,user_id_,distribution_column="popularity",p_g_u_all_users=p_g_u_all_users,)) for user_id_ in set(test["user"])]

    cg_genre_queue = [(user_id_, metricsInstance.tradeoff_genre_count(all_genres,train,dataset.items,user_id_,distribution_column="genres",p_g_u_all_users=p_g_u_genre_all_users,)) for user_id_ in set(test["user"])]
    var_genre_queue = [(user_id_, metricsInstance.tradeoff_variance(all_genres,train,dataset.items,user_id_,distribution_column="genres",p_g_u_all_users=p_g_u_genre_all_users,)) for user_id_ in set(test["user"])]
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

    # Criação do BPR Classico
    bpr_params = {
        'reg': 0.0001,
        'learning_rate': 0.01,
        'n_iters': 50,
        'n_factors': 15,
        'batch_size': 32,
    }
    bpr = BPR_classic(**bpr_params)
    model_name_list.append(f"BPR Classic")
    model_list.append(bpr)

    # Criação do BPR ideia 2
    bpr_params = {
        'reg': 0.0001,
        'learning_rate': 0.01,
        'n_iters': 50,
        'n_factors': 15,
        'batch_size': 32,
        'tradeoff': tradeoff,
        'distribution_column': 'genre',
        'p_t_u_all_users': p_g_u_genre_all_users,
        'p_t_i_all_items': p_t_i_genre_all_items,
        'p_t_u_pop_all_users': p_g_u_all_users,
        'p_t_i_pop_all_items': p_t_i_all_items,
        'dataset': trainratings,
        'movies_data': dataset.items,
        'tipo': 'gs1'
    }
    bpr = BPR_ideias(**bpr_params)
    model_name_list.append(f"BPR-Genre | Ideia 2 ")
    model_list.append(bpr)

    # Criação do BPR ideia 3
    bpr_params = {
        'reg': 0.0001,
        'learning_rate': 0.01,
        'n_iters': 50,
        'n_factors': 15,
        'batch_size': 32,
        'tradeoff': tradeoff,
        'distribution_column': 'genre',
        'p_t_u_all_users': p_g_u_genre_all_users,
        'p_t_i_all_items': p_t_i_genre_all_items,
        'p_t_u_pop_all_users': p_g_u_all_users,
        'p_t_i_pop_all_items': p_t_i_all_items,
        'dataset': trainratings,
        'movies_data': dataset.items,
        'tipo': 'pgs1'
    }
    bpr = BPR_ideias(**bpr_params)
    model_name_list.append(f"BPR-Genre | Ideia 3 ")
    model_list.append(bpr)

    # Criação do BPR ideia 4
    bpr_params = {
        'reg': 0.0001,
        'learning_rate': 0.01,
        'n_iters': 50,
        'n_factors': 15,
        'batch_size': 32,
        'tradeoff': tradeoff,
        'distribution_column': 'genre',
        'p_t_u_all_users': p_g_u_genre_all_users,
        'p_t_i_all_items': p_t_i_genre_all_items,
        'p_t_u_pop_all_users': p_g_u_all_users,
        'p_t_i_pop_all_items': p_t_i_all_items,
        'dataset': trainratings,
        'movies_data': dataset.items,
        'tipo': 'ps2'
    }
    bpr = BPR_ideias(**bpr_params)
    model_name_list.append(f"BPR-Genre | Ideia 4 ")
    model_list.append(bpr)

    # Criação do BPR ideia 5
    bpr_params = {
        'reg': 0.0001,
        'learning_rate': 0.01,
        'n_iters': 50,
        'n_factors': 15,
        'batch_size': 32,
        'tradeoff': tradeoff,
        'distribution_column': 'genre',
        'p_t_u_all_users': p_g_u_genre_all_users,
        'p_t_i_all_items': p_t_i_genre_all_items,
        'p_t_u_pop_all_users': p_g_u_all_users,
        'p_t_i_pop_all_items': p_t_i_all_items,
        'dataset': trainratings,
        'movies_data': dataset.items,
        'tipo': 'diff'
    }
    bpr = BPR_ideias(**bpr_params)
    model_name_list.append(f"BPR-Genre | Ideia 5 ")
    model_list.append(bpr)

    for model_name, model in zip(model_name_list, model_list):
        model.fit(trainset)
        
        print("MODEL FITTED, STARTING EX ERIMENT")
        started = time.time()
        exp = Pool(4)
        f = partial(
            run_user_bpr, train, dataset,
            calibration_column_list,
            model,
            trainratings,
            p_g_u_genre_all_users,
            p_g_u_all_users,
            p_t_i_genre_all_items,
            p_t_i_all_items,
            var_genre_all_users,
            cg_genre_all_users,
            var_all_users,
            cg_all_users,
            ratios_division,
            tradeoff,
            'kl'
        )
        print(f"Starting map. Elapsed Time: {time.time() - started}")
        exp_results = exp.map(
            f, set(test["user"])
        )
        exp.close()
        exp.join()
        print(f"Ending experiment. Elapsed Time: {time.time() - started}")
        
        for calibration_column in calibration_column_list:
            met = Pool(7)
            f = partial(
                calculate_tradeoff_metrics,
                model_name, int(indiceee),
                calibration_column,
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
                None,
                None,
                None,
                None,
                None,
                False
            )
            joint_tradeoffs = [tradeoff]

            met_results = met.map(
                f, joint_tradeoffs
            )
            met.close()
            met.join()
            for metrics_df in met_results:
                metrics_data.append(metrics_df)
        
        del model
        gc.collect()
        pd.DataFrame(metrics_data).to_csv(f"./results/yahoo_movies/bpr_tmp_{model_name}.csv")

    df = pd.concat([df, pd.DataFrame(metrics_data)])
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the experiment.')

    parser.add_argument(
        '--name',
        type=str,
        help='Name of the experiment',
        default=str(time.time())
    )

    args = parser.parse_args()

    dataset = MLDataset()
    # dataset.load_local_movielens_dataset("./datasets/ml-20m", type="ml20m_splitted", index=indiceee)
    dataset.load_local_movielens_dataset("./datasets/yahoo_movies", type='yahoo_new')

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

        for fold in [1, 2, 3]:
            for tradeoff in [0.0, 0.25, 0.5, 0.75, 1.0]:
                indiceee = fold

                df = pd.DataFrame([])

                model_list = []
                model_name_list = []

                df = run_experiment(model_name_list, model_list, dataset, df, tradeoff, calibration_column_list=['top10'])

                df.columns=[
                        'Model', "Fold", "Calibration Column",
                        "Tradeoff", "MACE Genres", "MACE Pop",
                        "LTC",  "MRMC Genres", "MRMC Pop",
                        "AGGDIV", "Prec@10", "MAP@10", "MRR",
                        "GAPBB", "GAPN", "GAPD"
                        ]
                df.reset_index()

                df.to_csv(f"./results/yahoo_movies/bpr_{time.time()}_{tradeoff}_{fold}_complete_index.csv")