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
from source.rerank.rerank_algo import run_user
from source.metrics.metrics import Metrics
import pandas as pd
import argparse
import time
from tqdm import tqdm
import multiprocessing
import gc
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.slope_one import SlopeOne
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.prediction_algorithms.matrix_factorization import SVDpp
from surprise.prediction_algorithms.matrix_factorization import NMF
from baselines.vae.splitters import min_rating_filter_pandas
from baselines.vae.splitters import numpy_stratified_split
from baselines.vae.sparse import AffinityMatrix
from baselines.vae.vae_utils import binarize
from multiprocessing import Pool
from functools import partial
import numpy as np

HELDOUT_USERS = 600 
SEED = 1

# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '1m'

# Model parameters
INTERMEDIATE_DIM = 1
LATENT_DIM = 1
EPOCHS = 1
BATCH_SIZE = 1

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

def run_experiment(model_name_list, model_list, dataset, df, calibration_column_list=["genres"]):
    metrics_data = []
    all_genres = set()

    for i in dataset.items['genres']:
        for genre in i.split("|"):
            all_genres.add(genre)
    print(all_genres)

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
        
        for model_name, model in zip(model_name_list, model_list):
            model.fit(x_train=train_data, 
                    x_valid=val_data, 
                    x_val_tr=val_data_tr, 
                    x_val_te=val_data_te_ratings, 
                    mapper=am_val)

            print("MODEL FITTED, STARTING EXPERIMENT")
            started = time.time()
            exp = Pool(4)
            f = partial(
                run_user, train, dataset,
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
                ratio_mean,
                'base'
            )
            #from itertools import islice
            #exp_results = exp.map(f, list(islice(test["user"], 1)))
            exp_results = exp.map(f, set(test["user"]))
            exp.close()
            exp.join()
            print(f"Ending experiment. Elapsed Time: {time.time() - started}")

            for calibration_column in calibration_column_list:
                met = Pool(7)
                f = partial(
                    calculate_tradeoff_metrics,
                    model_name, ind,
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
                joint_tradeoffs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

                met_results = met.map(
                    f, joint_tradeoffs
                )
                met.close()
                met.join()
                for metrics_df in met_results:
                    metrics_data.append(metrics_df)
            
            del model
            gc.collect()
            pd.DataFrame(metrics_data).to_csv(f"./results/ml-20m/genres_vae_tmp_{model_name}.csv")
    
    df = pd.concat([df, pd.DataFrame(metrics_data)])
    return df

if __name__ == '__main__':
    indiceee = '3'

    parser = argparse.ArgumentParser(description='Run the experiment.')

    parser.add_argument(
        '--name',
        type=str,
        help='Name of the experiment',
        default=str(time.time())
    )

    args = parser.parse_args()

    # dataset = MLDataset()
    # dataset.load_local_movielens_dataset("./datasets/ml-20m", type="ml20m_splitted", index=indiceee)
    # dataset.load_local_movielens_dataset("./datasets/yahoo_movies", type='vae')

    items_path = f"./datasets/ml-20m/items.csv"
    items = pd.read_csv(items_path, sep=',')
    items.columns = ['item', 'title', 'genres']

    #ratingsBase = pd.read_csv(f"./datasets/yahoo_movies/ratings.csv")
    #ratingsBase.columns = ["user", "item", "rating"]
    #ratings = ratingsBase.sample(frac=0.16, random_state=int(42))
    ratings = pd.read_csv(f"./datasets/ml-20m/ratings.csv")

    df_preferred = ratings[ratings['rating'] > 3.5]
    df_low_rating = ratings[ratings['rating'] <= 3.5]

    # Keep users who clicked on at least 5 movies
    dataset = min_rating_filter_pandas(df_preferred, min_rating=5, filter_by="user")

    # Keep movies that were clicked on by at least on 1 user
    dataset = min_rating_filter_pandas(dataset, min_rating=1, filter_by="item")

    # Obtain both usercount and itemcount after filtering
    usercount = dataset[['user']].groupby('user', as_index = False).size()
    itemcount = dataset[['item']].groupby('item', as_index = False).size()

    # Compute sparsity after filtering
    sparsity = 1. * dataset.shape[0] / (usercount.shape[0] * itemcount.shape[0])

    print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" % 
        (dataset.shape[0], usercount.shape[0], itemcount.shape[0], sparsity * 100))

    ratings = dataset

    items = pd.read_csv(items_path, sep=',')
    items.columns = ['item', 'title', 'genres']   

    unique_users = sorted(dataset.user.unique())
    np.random.seed(SEED)
    unique_users = np.random.permutation(unique_users)

    # Create train/validation/test users
    n_users = len(unique_users)
    print("Number of unique users:", n_users)

    train_users = unique_users[:(n_users - HELDOUT_USERS * 2)]
    print("\nNumber of training users:", len(train_users))

    val_users = unique_users[(n_users - HELDOUT_USERS * 2) : (n_users - HELDOUT_USERS)]
    print("\nNumber of validation users:", len(val_users))

    test_users = unique_users[(n_users - HELDOUT_USERS):]
    print("\nNumber of test users:", len(test_users))

    # For training set keep only users that are in train_users list
    train_set = dataset.loc[dataset['user'].isin(train_users)]
    print("Number of training observations: ", train_set.shape[0])

    # For validation set keep only users that are in val_users list
    val_set = dataset.loc[dataset['user'].isin(val_users)]
    print("\nNumber of validation observations: ", val_set.shape[0])

    # For test set keep only users that are in test_users list
    test_set = dataset.loc[dataset['user'].isin(test_users)]
    print("\nNumber of test observations: ", test_set.shape[0])

    # Obtain list of unique movies used in training set
    unique_train_items = pd.unique(train_set['item'])
    print("Number of unique movies that rated in training set", unique_train_items.size)

    mlDataset = MLDataset()
    mlDataset.train = train_set
    mlDataset.items = items
    mlDataset.test = test_set

    mlDataset = Popularity.generate_popularity_groups(mlDataset, subdivisions="mean", division='pareto')
    qnt_users = len(train_users)
    popularity_items = {}
    popularity_all = {}
    for item in unique_train_items:
        popularity_items[item] = len(mlDataset.train[mlDataset.train['item'] == item])/qnt_users
        popularity_all[item] = np.log2(len(mlDataset.train[mlDataset.train['item'] == item]))

    # For validation set keep only movies that used in training set
    val_set = val_set.loc[val_set['item'].isin(unique_train_items)]
    print("Number of validation observations after filtering: ", val_set.shape[0])

    # For test set keep only movies that used in training set
    test_set = test_set.loc[test_set['item'].isin(unique_train_items)]
    print("\nNumber of test observations after filtering: ", test_set.shape[0])

    # train_set/val_set/test_set contain user - movie interactions with rating 4 or 5

    # Instantiate the sparse matrix generation for train, validation and test sets
    # use list of unique items from training set for all sets
    am_train = AffinityMatrix(df=train_set, items_list=unique_train_items)
    am_val = AffinityMatrix(df=val_set, items_list=unique_train_items)
    am_test = AffinityMatrix(df=test_set, items_list=unique_train_items)

    # Obtain the sparse matrix for train, validation and test sets
    train_data, _, _ = am_train.gen_affinity_matrix()
    print(train_data.shape)

    val_data, val_map_users, val_map_items = am_val.gen_affinity_matrix()
    print(val_data.shape)

    test_data, test_map_users, test_map_items = am_test.gen_affinity_matrix()
    print(test_data.shape)

    # Split validation and test data into training and testing parts
    val_data_tr, val_data_te = numpy_stratified_split(val_data, ratio=0.75, seed=SEED)
    test_data_tr, test_data_te = numpy_stratified_split(test_data, ratio=0.75, seed=SEED)

    # Binarize train, validation and test data
    train_data = binarize(a=train_data, threshold=3.5)
    val_data = binarize(a=val_data, threshold=3.5)
    test_data = binarize(a=test_data, threshold=3.5)

    # Binarize validation data: training part  
    val_data_tr = binarize(a=val_data_tr, threshold=3.5)

    # Binarize validation data: testing part (save non-binary version in the separate object, will be used for calculating NDCG)
    val_data_te_ratings = val_data_te.copy()
    val_data_te = binarize(a=val_data_te, threshold=3.5)

    # Binarize test data: training part 
    test_data_tr = binarize(a=test_data_tr, threshold=3.5)

    # Binarize test data: testing part (save non-binary version in the separate object, will be used for calculating NDCG)
    test_data_te_ratings = test_data_te.copy()
    test_data_te = binarize(a=test_data_te, threshold=3.5)

    # retrieve real ratings from initial dataset 
    test_data_te_ratings=pd.DataFrame(test_data_te_ratings)
    val_data_te_ratings=pd.DataFrame(val_data_te_ratings)

    for index,i in df_low_rating.iterrows():
        user_old= i['user'] # old value 
        item_old=i['item'] # old value 

        if (test_map_users.get(user_old) is not None)  and (test_map_items.get(item_old) is not None) :
            user_new=test_map_users.get(user_old) # new value 
            item_new=test_map_items.get(item_old) # new value 
            rating=i['rating'] 
            test_data_te_ratings.at[user_new,item_new]= rating   

        if (val_map_users.get(user_old) is not None)  and (val_map_items.get(item_old) is not None) :
            user_new=val_map_users.get(user_old) # new value 
            item_new=val_map_items.get(item_old) # new value 
            rating=i['rating'] 
            val_data_te_ratings.at[user_new,item_new]= rating   

    val_data_te_ratings=val_data_te_ratings.to_numpy()    
    test_data_te_ratings=test_data_te_ratings.to_numpy()    
    # test_data_te_ratings  

    # Just checking
    print(np.sum(val_data))
    print(np.sum(val_data_tr))
    print(np.sum(val_data_te))

    # Just checking
    print(np.sum(test_data))
    print(np.sum(test_data_tr))
    print(np.sum(test_data_te))

    if mlDataset is not None:
        f = partial(calc_user_ratio, mlDataset)
        pool = Pool(multiprocessing.cpu_count()-3)
        aux = pool.map(
            f, train_users
        )
        pool.close()
        pool.join()

        f = partial(calc_user_ratio2, mlDataset)
        pool = Pool(multiprocessing.cpu_count()-3)
        aux2 = pool.map(
            f, train_users
        )
        pool.close()
        pool.join()

        ratios_division = {user : v for user, v in aux}
        ratio_mean= sum(v for user, v in aux)

        ratio_mean = ratio_mean/len(train_users)

        f = partial(calc_user_profile, mlDataset)
        pool = Pool(multiprocessing.cpu_count()-3)
        aux3 = pool.map(
            f, test_users
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

        models = []
        models_names = []

        from baselines.vae.modelos import VAE
        from baselines.vae.mult_vae import Mult_VAE
        #vae = VAE("vae_ml_1.json") # ou vae_yahoo_1.json pro dataset yahoo
        vae = Mult_VAE(n_users=train_data.shape[0], # Number of unique users in the training set
            original_dim=train_data.shape[1], # Number of unique items in the training set
            intermediate_dim=INTERMEDIATE_DIM,
            latent_dim=LATENT_DIM,
            n_epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            k=TOP_K,
            verbose=0,
            seed=SEED,
            save_path=None,
            drop_encoder=0.5,
            drop_decoder=0.5,
            annealing=False, 
        )
        models_names.append("VAE Explicity")
        models.append(vae)

        df = run_experiment(models_names, models, mlDataset, df, calibration_column_list=['genre'])

        df.columns=[
                'Model', "Fold", "Calibration Column",
                "Tradeoff", "MACE Genres", "MACE Pop",
                "LTC",  "MRMC Genres", "MRMC Pop",
                "AGGDIV", "Prec@10", "MAP@10", "MRR",
                "GAPBB", "GAPN", "GAPD"
                ]
        df.reset_index()

        df.to_csv(f"./results/ml-20m/genres_vae_{time.time()}_complete_index1.csv")