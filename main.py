import sys
import os
import pandas as pd
import numpy as np
import argparse
import time
import multiprocessing
import gc
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

from bpr_files.bpr_calibrated import BPR as BPR_ideia1
from bpr_files.bpr import BPR as BPR_classic
from dataset import Dataset as MLDataset
from metrics_calculation import calculate_tradeoff_metrics
from popularity_calculation import Popularity
from surprise.reader import Reader
from surprise import Dataset
from rerank_algo import run_rerank
from rerank_algo import run_bpr_rerank
from rerank_algo import run_in_processing_rerank
from metrics import Metrics
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.slope_one import SlopeOne
from surprise.prediction_algorithms.matrix_factorization import SVDpp
from surprise.prediction_algorithms.matrix_factorization import NMF
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from itertools import islice

def calculate_user_popularity_ratio(dataset, user_id):
    interacted_items = dataset.train[dataset.train['user'] == user_id]['item']
    relevant_items = dataset.items[dataset.items['item'].isin(interacted_items.values)]
    high_popularity_items = relevant_items[relevant_items['popularity'] == "H"]

    user_ratio = high_popularity_items.count()['popularity'] / len(interacted_items)
    return str(user_id), user_ratio

def get_user_profile(dataset, user_id):
    interacted_items = dataset.train[dataset.train['user'] == user_id].sort_values(by='item', ascending=False)
    high_rating_items = interacted_items[interacted_items['rating'] > 3]['item'].tolist()[:10]

    return user_id, high_rating_items

def categorize_user_by_popularity(dataset, user_id):
    BB_group = []
    N_group = []
    D_group = []

    interacted_items = dataset.train[dataset.train['user'] == user_id]['item']
    relevant_items = dataset.items[dataset.items['item'].isin(interacted_items.values)]
    high_popularity_items = relevant_items[relevant_items['popularity'] == "H"]
    low_popularity_items = relevant_items[relevant_items['popularity'] == "T"]

    user_ratio = high_popularity_items.count()['popularity'] / len(interacted_items)
    
    if user_ratio > 0.5:
        BB_group.append(user_id)
    elif low_popularity_items.count()['popularity'] / len(interacted_items) > 0.5:
        N_group.append(user_id)
    else:
        D_group.append(user_id)

    return BB_group, N_group, D_group

def calculate_target_distribution(dataset, distribution_column):
    f = partial(metrics_instance.calculate_target_item_distribution, dataset.items, distribution_column)

    pool = Pool(cpu_count() - 3)
    target_distribution = pool.map(f, set(dataset.train["item"]))
    pool.close()
    pool.join()

    return {id_: val for id_, val in target_distribution}

def calculate_target_user_distribution(train, dataset, distribution_column, rating_column, target_distribution):
    f = partial(
        metrics_instance.calculate_target_user_distribution,
        train, dataset.items, distribution_column, rating_column, target_distribution
    )

    pool = Pool(cpu_count() - 3)
    target_user_distribution = pool.map(f, set(dataset.test["user"]))
    pool.close()
    pool.join()

    return {id_: val for id_, val in target_user_distribution}

def run_pairwise(dataset, df, tradeoff):
    metrics_data = []

    dataset.train['user' + '_original'] = dataset.train['user'] 
    dataset.train['item' + '_original'] = dataset.train['item'] 
    dataset.train['user'] = dataset.train['user'].astype('category').cat.codes 
    dataset.train['item'] = dataset.train['item'].astype('category').cat.codes

    dataset.test['user' + '_original'] = dataset.test['user']
    dataset.test['item' + '_original'] = dataset.test['item']
    dataset.test['user'] = dataset.test['user'].astype('category').cat.codes
    dataset.test['item'] = dataset.test['item'].astype('category').cat.codes

    newDataset = dataset.items
    newDataset['item' + '_original'] = newDataset['item']
    newDataset = newDataset.merge(dataset.train[['item', 'item_original']], on = 'item_original', how = 'left')
    newDataset['item'] = newDataset['item_y']

    train = dataset.train
    test = dataset.test

    print("Calculating items distribution")

    started = time.time()
    target_all_items_distribution = calculate_target_distribution(dataset, "popularity")
    target_all_items_genres_distribution = calculate_target_distribution(dataset, "genres")
    print(f"Ending items distribution calculation. Elapsed Time: {time.time() - started}")

    print("Calculating users distribution")

    start = time.time()
    target_all_users_distribution = calculate_target_user_distribution(train, dataset, "popularity", "rating", target_all_items_distribution)
    target_all_users_genres_distribution = calculate_target_user_distribution(train, dataset, "genres", "rating", target_all_items_genres_distribution)
    print(f"Ending users distribution calculation. Elapsed Time: {time.time() - start}")
    print(f"Genre Finish: {time.time() - start}")
    
    trainratings = train

    print("MODEL FITTED, STARTING EXPERIMENT")
    started = time.time()
    exp = Pool(4)

    users = list(np.unique(dataset.train['user'].values))
    items = list(np.unique(dataset.train['item'].values))

    f = partial(run_in_processing_rerank, train, test, newDataset, users, items)
    exp_results = exp.map(f, set(test["user"]))
    #exp_results = exp.map(f, list(islice(test["user"], 1)))
    exp.close()
    exp.join()
    print(f"Ending experiment. Elapsed Time: {time.time() - started}")

    met = Pool(7)
    f = partial(
        calculate_tradeoff_metrics,
        "pairwise", fold,
        "inprocessing",
        target_all_users_genres_distribution,
        target_all_items_genres_distribution,
        newDataset,
        trainratings,
        target_all_users_distribution,
        target_all_items_distribution,
        test,
        exp_results,
        BB_group, N_group, D_group,
        popularity_items
    )

    met_results = met.map(f, [tradeoff])
    met.close()
    met.join()
    for metrics_df in met_results:
        metrics_data.append(metrics_df)
    
    gc.collect()
    pd.DataFrame(metrics_data).to_csv(f"./results/{selected_dataset}_{calibration_type}_fold_{fold}_tmp_pairwise.csv")
    
    df = pd.concat([df, pd.DataFrame(metrics_data)])
    return df


def run_experiment(model_name_list, model_list, dataset, df, calibration_type, tradeoff = 0.0):
    metrics_data = []
    all_genres = set(dataset.items['genres'].explode().unique())

    print(all_genres)

    train = dataset.train
    test = dataset.test
    reader = Reader()

    trainset = Dataset.load_from_df(train[["user", "item", "rating"]], reader = reader).build_full_trainset()
    
    print("Calculating items distribution")

    started = time.time()
    target_all_items_distribution = calculate_target_distribution(dataset, "popularity")
    target_all_items_genres_distribution = calculate_target_distribution(dataset, "genres")
    print(f"Ending items distribution calculation. Elapsed Time: {time.time() - started}")

    print("Calculating users distribution")

    start = time.time()
    target_all_users_distribution = calculate_target_user_distribution(train, dataset, "popularity", "rating", target_all_items_distribution)
    target_all_users_genres_distribution = calculate_target_user_distribution(train, dataset, "genres", "rating", target_all_items_genres_distribution)
    print(f"Ending users distribution calculation. Elapsed Time: {time.time() - start}")
    
    print(f"Genre Finish: {time.time() - start}")
    print("Calculating Users Custom Tradeoff")
    tradeoff_pool = Pool(multiprocessing.cpu_count()-3)
    started = time.time()
    cg_queue = [(user_id_, metrics_instance.tradeoff_genre_count(
        ["H", "M", "T"],
        train,
        dataset.items,
        user_id_,
        distribution_column = "popularity",
        target_all_users_distribution = target_all_users_distribution)) for user_id_ in set(test["user"])]
    var_queue = [(user_id_, metrics_instance.tradeoff_variance(
        ["H", "M", "T"],
        train,
        dataset.items,
        user_id_,
        distribution_column = "popularity",
        target_all_users_distribution = target_all_users_distribution,)) for user_id_ in set(test["user"])]

    cg_genre_queue = [(user_id_, metrics_instance.tradeoff_genre_count(
        all_genres,
        train,
        dataset.items,
        user_id_,
        distribution_column = "genres",
        target_all_users_distribution = target_all_users_genres_distribution,)) for user_id_ in set(test["user"])]
    var_genre_queue = [(user_id_, metrics_instance.tradeoff_variance(
        all_genres,
        train,
        dataset.items,
        user_id_,
        distribution_column = "genres",
        target_all_users_distribution = target_all_users_genres_distribution,)) for user_id_ in set(test["user"])]
    
    print(f"[P1] Terminei de montar as listas {time.time() - started}")
    tradeoff_pool.close()
    tradeoff_pool.join()
    cg_all_users = {user_id_: result_async for user_id_, result_async in tqdm(cg_queue)}
    var_all_users = {user_id_: result_async for user_id_, result_async in tqdm(var_queue)}
    cg_genre_all_users = {user_id_: result_async for user_id_, result_async in tqdm(cg_genre_queue)}
    var_genre_all_users = {user_id_: result_async for user_id_, result_async in tqdm(var_genre_queue)}
    print(f"Ending users custom tradeoffs. Elapsed Time: {time.time() - started}")
    
    trainratings = train

    if (calibration_type_value == "7"):
        bpr_params = {
            'distribution_column': 'popularity',
            'target_all_users_distribution': target_all_users_genres_distribution,
            'target_all_items_distribution': target_all_items_genres_distribution,
            'dataset': trainratings,
            'movies_data': dataset.items
        }
        bpr = BPR_ideia1(**bpr_params)
        model_name_list.append(f"BPR-Popularity | Ideia 1 ")
        model_list.append(bpr)
        
    for model_name, model in zip(model_name_list, model_list):
        model.fit(trainset)

        print("MODEL FITTED, STARTING EXPERIMENT")
        started = time.time()
        exp = Pool(4)

        if (calibration_type_value == "6" or calibration_type_value == "7"):
            f = partial(run_bpr_rerank, train, dataset, model, tradeoff)
        else:        
            f = partial(
                run_rerank, train, dataset,
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
                calibration_calculation_type
            )        
        
        #exp_results = exp.map(f, list(islice(test["user"], 1)))
        exp_results = exp.map(f, set(test["user"]))
        exp.close()
        exp.join()
        print(f"Ending experiment. Elapsed Time: {time.time() - started}")

        met = Pool(7)
        f = partial(
            calculate_tradeoff_metrics,
            model_name, fold,
            calibration_type,
            target_all_users_genres_distribution,
            target_all_items_genres_distribution,
            dataset,
            trainratings,
            target_all_users_distribution,
            target_all_items_distribution,
            test,
            exp_results,
            BB_group, N_group, D_group,
            popularity_items
        )

        if (calibration_type_value == "6" or calibration_type_value == "7"):
            joint_tradeoffs = [tradeoff]
        else:
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
        pd.DataFrame(metrics_data).to_csv(f"./results/{selected_dataset}_{calibration_type}_fold_{fold}_tmp_{model_name}.csv")

    df = pd.concat([df, pd.DataFrame(metrics_data)])
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run the experiment.')

    parser.add_argument(
        '--name',
        type = str,
        help = 'Name of the experiment',
        default = str(time.time())
    )

    args = parser.parse_args()
    fold = input("Enter the fold number: ")
    calibration_calculation_type = "kl"

    print("Type the system calibration type:")
    print("1. Genres (Steck)")
    print("2. Popularity (Abdoullahpouri)")
    print("3. Personalized")
    print("4. Popularity")
    print("5. Two Stage")
    print("6. BPR")
    print("7. Calibrated BPR")
    print("8. Pairwise")

    valid_options = ["1", "2", "3", "4", "5", "6", "7", "8"]

    while True:
        calibration_type = input("Enter the number of the desired calibration type: ")
        calibration_type_value = calibration_type
        if calibration_type in valid_options:
            break
        else:
            print("Invalid option. Please enter a valid number.")         

    if calibration_type in valid_options:
        if calibration_type == "1":
            calibration_type = "genre"
        elif calibration_type == "2":
            calibration_type = "popularity"
            calibration_calculation_type = "jansen"
        elif calibration_type == "3":
            calibration_type = "personalized"
        elif calibration_type == "4":
            calibration_type = "popularity"
        elif calibration_type == "5":
            calibration_type = "double"
        elif calibration_type == "6":
            calibration_type = "top10"
        elif calibration_type == "7":
            calibration_type = "top10"
        elif calibration_type == "8":
            calibration_type = "inprocessing"
        print(f"You chose the calibration type: {calibration_type}")

    print("Type the dataset:")
    print("1. Yahoo Movies")
    print("2. Yahoo Songs")
    print("3. Movielens")

    valid_dataset_options = ["1", "2", "3"]

    while True:
        selected_dataset = input("Enter the number of the dataset: ")
        if selected_dataset in valid_dataset_options:
            break
        else:
            print("Invalid option. Please enter a valid number.")   

    dataset = MLDataset()
    if selected_dataset in valid_dataset_options:
        if selected_dataset == "1":
            selected_dataset = "yahoo_movies"
            dataset.load_dataset("./datasets/yahoo_movies", type = "yahoo_movies")
        elif selected_dataset == "2":
            selected_dataset = "yahoo_songs"
            dataset.load_dataset("./datasets/yahoo_song", type = "yahoo_song")
        elif selected_dataset == "3":
            selected_dataset = "movielens"
            dataset.load_dataset("./datasets/ml-20m", type = "movielens")

    print("Dados Train")
    print(dataset.train.shape)
    print(len(dataset.train['item'].unique().tolist()))
    print(len(dataset.train['user'].unique().tolist()))

    print("Dados Test")
    print(dataset.test.shape)
    print(len(dataset.test['item'].unique().tolist()))
    print(len(dataset.test['user'].unique().tolist()))

    dataset = Popularity.generate_popularity_groups(dataset)

    if dataset is not None:
        qnt_users = len(dataset.train['user'].unique().tolist())
        popularity_items = {}
        popularity_all = {}
        for item in dataset.train['item'].unique().tolist():
            popularity_items[item] = len(dataset.train[dataset.train['item'] == item])/qnt_users
            popularity_all[item] = np.log2(len(dataset.train[dataset.train['item'] == item]))

        f = partial(calculate_user_popularity_ratio, dataset)
        pool = Pool(multiprocessing.cpu_count()-3)
        aux = pool.map(f, dataset.train['user'].unique())
        pool.close()
        pool.join()

        f = partial(categorize_user_by_popularity, dataset)
        pool = Pool(multiprocessing.cpu_count()-3)
        aux2 = pool.map(f, dataset.train['user'].unique())
        pool.close()
        pool.join()

        ratios_division = {user : v for user, v in aux}
        ratio_mean= sum(v for _, v in aux)
        ratio_mean = ratio_mean/len(dataset.train['user'].unique())

        f = partial(get_user_profile, dataset)
        pool = Pool(multiprocessing.cpu_count()-3)
        aux3 = pool.map(f, dataset.train['user'].unique())
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

        models = []
        models_names = []

        if (calibration_type_value == "1" or calibration_type_value == "2" or calibration_type_value == "3" or 
            calibration_type_value == "4" or calibration_type_value == "5"):
            sim_options = {"name": "pearson_baseline", "user_based": False}
            itemknn = KNNWithMeans(k = 30, sim_options = sim_options)
            models.append(itemknn)
            models_names.append("itemknn")

            slope = SlopeOne()
            #models.append(slope)
            #models_names.append("slopeOne")

            svdpp = SVDpp(n_epochs = 20, n_factors = 20, lr_all = 0.005, reg_all = 0.02)
            #models.append(svdpp)
            #models_names.append("SVDpp")

            nmf = NMF(n_epochs = 50, n_factors = 15, reg_bu = 0.06, reg_bi = 0.06)
            #models.append(nmf)
            #models_names.append("NMF")
        elif (calibration_type_value == "6"):
            bpr = BPR_classic()
            models_names.append(f"BPR Classic")
            models.append(bpr)
        elif (calibration_type_value == "8"):
            models_names.append(f"pairwise")

        metrics_instance = Metrics()

        if (calibration_type_value == "8"):
            for tradeoff in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                df = pd.DataFrame([])
                df = run_pairwise(dataset, df, tradeoff)
                df.columns=[
                'Model', "Fold", "Calibration Column",
                "Tradeoff", "MACE Genres", "MACE Pop",
                "LTC",  "MRMC Genres", "MRMC Pop",
                "AGGDIV", "Prec@10", "MAP@10", "MRR",
                "GAPBB", "GAPN", "GAPD"
                ]
                df.reset_index()
                df.to_csv(f"./results/{selected_dataset}_{calibration_type}_fold_{fold}_{time.time()}_complete.csv")
        elif (calibration_type_value == "6" or calibration_type_value == "7"):
            for tradeoff in [0.0]:
                df = pd.DataFrame([])
                df = run_experiment(models_names, models, dataset, df, calibration_type, tradeoff)
                df.columns=[
                    'Model', "Fold", "Calibration Column",
                    "Tradeoff", "MACE Genres", "MACE Pop",
                    "LTC",  "MRMC Genres", "MRMC Pop",
                    "AGGDIV", "Prec@10", "MAP@10", "MRR",
                    "GAPBB", "GAPN", "GAPD"
                ]
                df.reset_index()
                df.to_csv(f"./results/{selected_dataset}_{calibration_type}_fold_{fold}_{time.time()}_complete.csv")
        else:
            df = pd.DataFrame([])
            df = run_experiment(models_names, models, dataset, df, calibration_type)
            df.columns=[
                'Model', "Fold", "Calibration Column",
                "Tradeoff", "MACE Genres", "MACE Pop",
                "LTC",  "MRMC Genres", "MRMC Pop",
                "AGGDIV", "Prec@10", "MAP@10", "MRR",
                "GAPBB", "GAPN", "GAPD"
                ]
            df.reset_index()
            df.to_csv(f"./results/{selected_dataset}_{calibration_type}_fold_{fold}_{time.time()}_complete.csv")
