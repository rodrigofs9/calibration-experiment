import sys
import os
import pandas as pd
import time
import multiprocessing
import gc
import argparse
import numpy as np

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

from source.dataset.dataset import Dataset as MLDataset
from source.metrics.metrics_calculation import calculate_tradeoff_metrics
from source.popularity.popularity import Popularity
from surprise.reader import Reader
from surprise import Dataset
from source.rerank.rerank_algo import run_user
from source.metrics.metrics import Metrics
from tqdm import tqdm
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.slope_one import SlopeOne
from surprise.prediction_algorithms.matrix_factorization import SVDpp, NMF
from multiprocessing import Pool
from functools import partial

def calculate_user_popularity_ratio(dataset, user_id):
    interacted_by_user = dataset.train[dataset.train['user'] == user_id]['item']
    popularities = dataset.items[dataset.items['item'].isin(interacted_by_user.values)]
    high_pop = popularities[popularities['popularity'] == "H"]

    return str(user_id), high_pop.count()['popularity']/len(list(interacted_by_user))


def calculate_user_profile(dataset, user_id, rating_threshold=3, profile_length=10):
    user_interactions = dataset.train[dataset.train['user'] == user_id]

    high_rating_items = user_interactions[user_interactions['rating'] > rating_threshold]

    user_profile = high_rating_items['item'].tolist()[:profile_length]

    return user_id, user_profile

def calculate_user_ratio_group(dataset, user_id):
    user_interactions = dataset.train[dataset.train['user'] == user_id]
    
    if user_interactions.empty:
        return [], [], []

    interacted_items = user_interactions['item']
    popularities = dataset.items[dataset.items['item'].isin(interacted_items.values)]
    
    high_pop_count = popularities[popularities['popularity'] == "H"].shape[0]
    low_pop_count = popularities[popularities['popularity'] == "T"].shape[0]
    total_interactions = len(interacted_items)
    
    BB_group, N_group, D_group = [], [], []

    if high_pop_count / total_interactions > 0.5:
        BB_group.append(user_id)
    elif low_pop_count / total_interactions > 0.5:
        N_group.append(user_id)
    else:
        D_group.append(user_id)

    return BB_group, N_group, D_group

def calculate_item_distribution(dataset, items, distribution_column):
    metrics_instance = Metrics()
    f = partial(metrics_instance.get_p_t_i_distribution_mp, dataset.items, distribution_column)
    
    pool = Pool(multiprocessing.cpu_count() - 3)
    p_t_i_values = pool.map(f, set(items))
    pool.close()
    pool.join()
    
    return {id_: val for id_, val in p_t_i_values}

def calculate_user_distribution(train, test, dataset, distribution_column, p_t_i_values):
    metrics_instance = Metrics()
    ptu_pool = Pool(multiprocessing.cpu_count() - 3)
    f = partial(metrics_instance.get_p_t_u_distribution_mp, train, dataset.items, distribution_column, "rating", p_t_i_values)
    p_g_u_values = ptu_pool.map(f, test)
    ptu_pool.close()
    ptu_pool.join()
    
    return {id_: val for id_, val in p_g_u_values}

def calculate_users_custom_tradeoff(all_genres, test, train, dataset, distribution_column, p_g_u_values):
    metrics_instance = Metrics()
    tradeoff_pool = Pool(multiprocessing.cpu_count() - 3)
    
    cg_queue = [(user_id, metrics_instance.tradeoff_genre_count(["H", "M", "T"], train, dataset.items, user_id, distribution_column, p_g_u_values)) for user_id in set(test["user"])]
    var_queue = [(user_id, metrics_instance.tradeoff_variance(["H", "M", "T"], train, dataset.items, user_id, distribution_column, p_g_u_values)) for user_id in set(test["user"])]

    cg_genre_queue = [(user_id, metrics_instance.tradeoff_genre_count(all_genres, train, dataset.items, user_id, "genres", p_g_u_values)) for user_id in set(test["user"])]
    var_genre_queue = [(user_id, metrics_instance.tradeoff_variance(all_genres, train, dataset.items, user_id, "genres", p_g_u_values)) for user_id in set(test["user"])]

    tradeoff_pool.close()
    tradeoff_pool.join()
    
    cg_all_users = {user_id: result_async for user_id, result_async in tqdm(cg_queue)}
    var_all_users = {user_id: result_async for user_id, result_async in tqdm(var_queue)}
    cg_genre_all_users = {user_id: result_async for user_id, result_async in tqdm(cg_genre_queue)}
    var_genre_all_users = {user_id: result_async for user_id, result_async in tqdm(var_genre_queue)}

    return cg_all_users, var_all_users, cg_genre_all_users, var_genre_all_users

def evaluate_model(models, dataset, df, calibration_column_list=["genres"]):
    metrics_data = []
    all_genres = set()
    
    for genres in dataset.items['genres']:
        for genre in genres.split("|"):
            all_genres.add(genre)
    print(all_genres)

    for fold in range(1,3):
        train = dataset.train
        test = dataset.test
        reader = Reader()

        trainset = Dataset.load_from_df(
            train[["user", "item", "rating"]], reader=reader
        ).build_full_trainset()
        
        print("Calculating items distribution")
        started = time.time()
        p_t_i_all_items = calculate_item_distribution(dataset, train["item"], "popularity")
        p_t_i_genre_all_items = calculate_item_distribution(dataset, train["item"], "genres")
        print(f"Ending items distribution calculation. Elapsed Time: {time.time() - started}")

        print("Calculating users distribution")
        start = time.time()
        p_g_u_all_users = calculate_user_distribution(train, set(test["user"]), dataset, "popularity", p_t_i_all_items)
        p_g_u_genre_all_users = calculate_user_distribution(train, set(test["user"]), dataset, "genres", p_t_i_genre_all_items)
        print(f"Ending users distribution calculation. Elapsed Time: {time.time() - start}")

        cg_all_users, var_all_users, cg_genre_all_users, var_genre_all_users = calculate_users_custom_tradeoff(all_genres, test, train, dataset, "popularity", p_g_u_all_users)
        
        trainratings = train

        for model_name, model in models:
            model.fit(trainset)

            exp_results = run_user_experiment(model, train, test, dataset, calibration_column_list, trainratings,
                                            p_g_u_genre_all_users, p_g_u_all_users, p_t_i_genre_all_items, p_t_i_all_items,
                                            var_genre_all_users, cg_genre_all_users, var_all_users, cg_all_users,
                                            ratios_division, ratio_mean)

            for calibration_column in calibration_column_list:
                met = Pool(7)
                f = partial(
                    calculate_tradeoff_metrics,
                    model_name, fold,
                    calibration_column,
                    p_g_u_genre_all_users,
                    p_t_i_genre_all_items,
                    dataset,
                    trainratings,
                    p_g_u_all_users,
                    p_t_i_all_items,
                    test,
                    exp_results,
                    blockbuster_users, niche_users,
                    diverse_users, item_popularity,
                    None,None,None,None,None,False
                )
                joint_tradeoffs = [0, 0.2, 0.4, 0.6, 0.8, 1]

                met_results = met.map(
                    f, joint_tradeoffs
                )
                met.close()
                met.join()
                for metrics_df in met_results:
                    metrics_data.append(metrics_df)

            del model
            gc.collect()
            pd.DataFrame(metrics_data).to_csv(f"./results/{dataset_name}/base_genres_tmp_{model_name}.csv")

        df = pd.concat([df, pd.DataFrame(metrics_data)])
        return df

def run_user_experiment(model, train, test, dataset, calibration_column_list, trainratings, p_g_u_genre_all_users,
                        p_g_u_all_users, p_t_i_genre_all_items, p_t_i_all_items, var_genre_all_users,
                        cg_genre_all_users, var_all_users, cg_all_users, ratios_division, ratio_mean):
    print("MODEL FITTED, STARTING EXPERIMENT")
    started = time.time()
    exp = Pool(4)
    f = partial(run_user, train, dataset, calibration_column_list, model, trainratings, p_g_u_genre_all_users,
                p_g_u_all_users, p_t_i_genre_all_items, p_t_i_all_items, var_genre_all_users, cg_genre_all_users,
                var_all_users, cg_all_users, ratios_division, ratio_mean, 'base')
    print(f"Starting map. Elapsed Time: {time.time() - started}")
    exp_results = exp.map(f, set(test["user"]))
    exp.close()
    exp.join()
    print(f"Ending experiment. Elapsed Time: {time.time() - started}")
    return exp_results

# Experiment Setup
timeValue = str(time.time())
dataset_name = 'yahoo_song'
dataset_type = 'yahoo_song_new'
calibration = 'genre'
num_cpu_cores = multiprocessing.cpu_count() - 3

def load_dataset():
    dataset = MLDataset()
    dataset.load_local_movielens_dataset(f"./datasets/{dataset_name}", type=dataset_type)
    return dataset

def evaluate_models_and_record_results(models, dataset):
    df = pd.DataFrame([])
    df = evaluate_model(models, dataset, df, calibration_column_list=[calibration])

    df.columns=[
            'Model', "Fold", "Calibration Column",
            "Tradeoff", "MACE Genres", "MACE Pop",
            "LTC",  "MRMC Genres", "MRMC Pop",
            "AGGDIV", "Prec@10", "MAP@10", "MRR",
            "GAPBB", "GAPN", "GAPD"
            ]
    df.reset_index()

    df.to_csv(f"./results/{dataset}/{calibration}_{time}_complete.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the experiment.')
    parser.add_argument('--name', type=str, help='Name of the experiment', default=timeValue)
    args = parser.parse_args()

    dataset = load_dataset()

    print("Train Data Size")
    print(dataset.train.shape)
    print(len(dataset.train['item'].unique().tolist()))
    print(len(dataset.train['user'].unique().tolist()))

    print("Test Data Size")
    print(dataset.test.shape)
    print(len(dataset.test['item'].unique().tolist()))
    print(len(dataset.test['user'].unique().tolist()))

    dataset = Popularity.generate_popularity_groups(dataset, subdivisions="mean", division='pareto')

    if dataset is not None:
        num_users = len(dataset.train['user'].unique())
        
        item_popularity = {}
        total_popularity = {}
        item_counts = dataset.train['item'].value_counts()
        for item in dataset.train['item'].unique():
            item_popularity[item] = item_counts[item] / num_users
            total_popularity[item] = np.log2(item_counts[item])

        pool = Pool(multiprocessing.cpu_count() - 3)

        calculate_user_popularity_ratio_func = partial(calculate_user_popularity_ratio, dataset)
        user_popularity_ratio_data = pool.map(calculate_user_popularity_ratio_func, dataset.train['user'].unique())

        calculate_user_ratio_group_func = partial(calculate_user_ratio_group, dataset)
        user_ratio_group_data = pool.map(calculate_user_ratio_group_func, dataset.train['user'].unique())

        calculate_user_profile_func = partial(calculate_user_profile, dataset)
        user_profile_data = pool.map(calculate_user_profile_func, dataset.train['user'].unique())

        ratios_division = dict(user_popularity_ratio_data)
        ratio_mean = sum(v for _, v in user_popularity_ratio_data)/num_users

        pool.close()
        pool.join()

        user_ratios = {user: ratio for user, ratio in user_popularity_ratio_data}
        user_profiles = {user: profile for user, profile in user_profile_data}

        blockbuster_users = [BB[0] for BB, N, D in user_ratio_group_data if BB]
        niche_users = [N[0] for BB, N, D in user_ratio_group_data if N]
        diverse_users = [D[0] for BB, N, D in user_ratio_group_data if D]

        print("Num of Blockbuster users (BB):", len(blockbuster_users))
        print("Num of Niche users (N):", len(niche_users))
        print("Num of Diverse users (D):", len(diverse_users))
        
        models = [
            ("itemknn", KNNWithMeans(k=5, sim_options={"name": "pearson_baseline", "user_based": False})),
            ("slopeOne", SlopeOne()),
            ("SVDpp", SVDpp(n_epochs=20, n_factors=20, lr_all=0.005, reg_all=0.02)),
            ("NMF", NMF(n_epochs=50, n_factors=15, reg_bu=0.06, reg_bi=0.06))
        ]

        calibration_columns = [calibration]
        evaluate_models_and_record_results(models, dataset)