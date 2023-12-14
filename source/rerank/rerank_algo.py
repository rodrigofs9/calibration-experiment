import pandas as pd
from surprise.reader import Reader
from sklearn.model_selection import train_test_split
from surprise import Trainset
from surprise import Dataset
from source.rerank.mmr import re_rank_list, re_rank_list2, re_rank_list_PPTM
import time
import numpy as np
from baselines.pairwise.pairwise import PairWise
reader = Reader()

def run_user_in_processing_mitigation(train, test, dataset, users, items, user, tradeoff = 0.0):
    recomend_results = {
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
    for index, row in items_metadata.iterrows():
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

        recomend_results['inprocessing'][0.0]['reranked'][user] = result_tuples

    return user, recomend_results

def run_user_biasmitigation(train, dataset,model, popularity_all,user):
    recomend_results = {'popbiasmitigation': {}}
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
    predictions = [
        (pred.iid, pred.est)
        for pred in predictions
        if pred.uid == user
    ]
    for alpha in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        if alpha not in recomend_results['popbiasmitigation']:
                recomend_results['popbiasmitigation'][alpha] = {"reranked": {}}

        new_predictions = [(id_, alpha*popularity_all[id_] + (1-alpha)*rat_) for id_, rat_ in predictions]

        rec_list = sorted(new_predictions, key=lambda x: x[1], reverse=True)
        recomend_results['popbiasmitigation'][alpha]["reranked"][user] = rec_list[:10]

    scores = {}
    users__ = []
    for user_id in set(users__):
        scores[user_id] = recomend_results

    return user, recomend_results

def run_user(
    train, dataset,
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
    calibration_type,
    user
):
    try:
        recomend_results = {}

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
        #aux_3 = sorted(
        #    [
        #        (pred[0], pred[1])
        #        for pred in predictions
        #        if pred[0] == user
        #    ],
        #    key=lambda x: x[1],
        #    reverse=True,
        #)
        aux_3 = sorted(
                    [
                        (pred.iid, pred.est)
                        for pred in predictions
                        if pred.uid == user
                    ],
                    key=lambda x: x[1],
                    reverse=True,
        )

        recomended_user = aux_3

        var_gc = {
            'genre': {'VAR': var_genre_all_users, "GC": cg_genre_all_users},
            'popularity': {'VAR': var_all_users, "GC": cg_all_users},
        }

        for calibration_column in calibration_column_list:
            if calibration_column not in recomend_results:
                recomend_results[calibration_column] = {}

            for tradeoff in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                if tradeoff not in recomend_results[calibration_column]:
                    recomend_results[calibration_column][tradeoff] = {"reranked": {}}

                tradeoff_value = tradeoff

                re_ranked = None
                if calibration_column == "genre":

                    if tradeoff == "VAR":
                        tradeoff_value = var_genre_all_users[user]
                    elif tradeoff == "GC":
                        tradeoff_value = cg_genre_all_users[user]
                    
                    re_ranked, _  = re_rank_list(
                    trainratings,
                    dataset.items,
                    user,
                    recomended_user[:100],
                    p_g_u_all_users=p_g_u_genre_all_users,
                    p_t_i_all_items=p_t_i_genre_all_items,
                    tradeoff=tradeoff_value,
                    N=10,
                    distribution_column="genres",
                    calibration_type=calibration_type
                )
                elif calibration_column == "double":
                    if tradeoff == "VAR":
                        tradeoff_value = var_genre_all_users[user]
                    elif tradeoff == "GC":
                        tradeoff_value = cg_genre_all_users[user]
                    
                    re_ranked, first_calibration  = re_rank_list(
                    trainratings,
                    dataset.items,
                    user,
                    recomended_user[:100],
                    p_g_u_all_users=p_g_u_genre_all_users,
                    p_t_i_all_items=p_t_i_genre_all_items,
                    tradeoff=tradeoff_value,
                    N=100,
                    distribution_column="popularity",
                    calibration_type=calibration_type
                    )
                    
                    re_ranked, _  = re_rank_list(
                    trainratings,
                    dataset.items,
                    user,
                    first_calibration,
                    p_g_u_all_users=p_g_u_genre_all_users,
                    p_t_i_all_items=p_t_i_genre_all_items,
                    tradeoff=tradeoff_value,
                    N=10,
                    distribution_column="genres",
                    calibration_type=calibration_type
                    )
                elif calibration_column == 'personalized':
                    if ratios_division[str(user)] >= ratio_mean :
                        if tradeoff == "VAR":
                            tradeoff_value = var_genre_all_users[user]
                        elif tradeoff == "GC":
                            tradeoff_value = cg_genre_all_users[user]
                        
                        re_ranked, _  = re_rank_list(
                        trainratings,
                        dataset.items,
                        user,
                        recomended_user[:100],
                        p_g_u_all_users=p_g_u_genre_all_users,
                        p_t_i_all_items=p_t_i_genre_all_items,
                        tradeoff=tradeoff_value,
                        N=10,
                        distribution_column="genres",
                        calibration_type=calibration_type
                    )
                    else:
                        if tradeoff == "VAR":
                            tradeoff_value = var_all_users[user]
                        elif tradeoff == "GC":
                            tradeoff_value = cg_all_users[user]

                        re_ranked, _  = re_rank_list(
                        trainratings,
                        dataset.items,
                        user,
                        recomended_user[:100],
                        p_g_u_all_users=p_g_u_all_users,
                        p_t_i_all_items=p_t_i_all_items,
                        tradeoff=tradeoff_value,
                        N=10,
                        distribution_column="popularity",
                        calibration_type=calibration_type
                    )
                elif calibration_column == 'inv_personalized':
                    if ratios_division[str(user)] <= ratio_mean :
                        if tradeoff == "VAR":
                            tradeoff_value = var_genre_all_users[user]
                        elif tradeoff == "GC":
                            tradeoff_value = cg_genre_all_users[user]
                        
                        re_ranked, _  = re_rank_list(
                        trainratings,
                        dataset.items,
                        user,
                        recomended_user[:100],
                        p_g_u_all_users=p_g_u_genre_all_users,
                        p_t_i_all_items=p_t_i_genre_all_items,
                        tradeoff=tradeoff_value,
                        N=10,
                        distribution_column="genres",
                        calibration_type=calibration_type
                    )
                    else:
                        if tradeoff == "VAR":
                            tradeoff_value = var_all_users[user]
                        elif tradeoff == "GC":
                            tradeoff_value = cg_all_users[user]

                        re_ranked, _  = re_rank_list(
                        trainratings,
                        dataset.items,
                        user,
                        recomended_user[:100],
                        p_g_u_all_users=p_g_u_all_users,
                        p_t_i_all_items=p_t_i_all_items,
                        tradeoff=tradeoff_value,
                        N=10,
                        distribution_column="popularity",
                        calibration_type=calibration_type
                    )
                else:

                    if tradeoff == "VAR":
                        tradeoff_value = var_all_users[user]
                        tradeoff_value_pop = var_all_users[user]
                    elif tradeoff == "GC":
                        tradeoff_value = cg_all_users[user]
                        tradeoff_value_pop = cg_all_users[user]

                    re_ranked, _  = re_rank_list(
                        trainratings,
                        dataset.items,
                        user,
                        recomended_user[:100],
                        p_g_u_all_users=p_g_u_all_users,
                        p_t_i_all_items=p_t_i_all_items,
                        tradeoff=tradeoff_value,
                        N=10,
                        distribution_column="popularity",
                        calibration_type=calibration_type
                    )
                
                recomend_results[calibration_column][tradeoff]["reranked"][user] = [
                    (item, index + 1)
                    for index, item in enumerate(re_ranked)
                ]
        print('teste loko')
        print(user)
        print(recomend_results)
        return user, recomend_results
    except Exception as e:
            print("An error occurred in run_user:")
            print(e)

def run_user_bpr(
    train, dataset,
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
    calibration_type,
    user
):
    try:
        recomend_results = {
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
            key=lambda x: x[1],
            reverse=True,
        )

        lista_ordenada = sorted(aux_3, key=lambda x: x[1], reverse=True)

        top_10 = [(item[0], i + 1) for i, item in enumerate(lista_ordenada[:10])]

        recomend_results['top10'][tradeoff]["reranked"][user] = [
                            item
                            for _, item in enumerate(top_10)
                        ]
        
        return user, recomend_results
    except Exception as e:
            print("An error occurred in run_user:")
            print(e)


def run_user_bpr_copy(
    train, dataset,
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
    calibration_type,
    user
):
    try:
        recomend_results = {}

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
                if pred[0] == user
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        #aux_3 = sorted(
        #            [
        #                (pred.iid, pred.est)
        #                for pred in predictions
        #                if pred.uid == user
        #            ],
        #            key=lambda x: x[1],
        #            reverse=True,
        #        )

        recomended_user = aux_3

        var_gc = {
            'genre': {'VAR': var_genre_all_users, "GC": cg_genre_all_users},
            'popularity': {'VAR': var_all_users, "GC": cg_all_users},
        }

        for calibration_column in calibration_column_list:
            if calibration_column not in recomend_results:
                recomend_results[calibration_column] = {}

            for tradeoff in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                if tradeoff not in recomend_results[calibration_column]:
                    recomend_results[calibration_column][tradeoff] = {"reranked": {}}

                tradeoff_value = tradeoff

                re_ranked = None
                if calibration_column == "genre":

                    if tradeoff == "VAR":
                        tradeoff_value = var_genre_all_users[user]
                    elif tradeoff == "GC":
                        tradeoff_value = cg_genre_all_users[user]
                    
                    re_ranked, _  = re_rank_list(
                    trainratings,
                    dataset.items,
                    user,
                    recomended_user[:100],
                    p_g_u_all_users=p_g_u_genre_all_users,
                    p_t_i_all_items=p_t_i_genre_all_items,
                    tradeoff=tradeoff_value,
                    N=10,
                    distribution_column="genres",
                    calibration_type=calibration_type
                )
                elif calibration_column == 'personalized':
                    if ratios_division[str(user)] >= ratio_mean :
                        if tradeoff == "VAR":
                            tradeoff_value = var_genre_all_users[user]
                        elif tradeoff == "GC":
                            tradeoff_value = cg_genre_all_users[user]
                        
                        re_ranked, _  = re_rank_list(
                        trainratings,
                        dataset.items,
                        user,
                        recomended_user[:100],
                        p_g_u_all_users=p_g_u_genre_all_users,
                        p_t_i_all_items=p_t_i_genre_all_items,
                        tradeoff=tradeoff_value,
                        N=10,
                        distribution_column="genres",
                        calibration_type=calibration_type
                    )
                    else:
                        if tradeoff == "VAR":
                            tradeoff_value = var_all_users[user]
                        elif tradeoff == "GC":
                            tradeoff_value = cg_all_users[user]

                        re_ranked, _  = re_rank_list(
                        trainratings,
                        dataset.items,
                        user,
                        recomended_user[:100],
                        p_g_u_all_users=p_g_u_all_users,
                        p_t_i_all_items=p_t_i_all_items,
                        tradeoff=tradeoff_value,
                        N=10,
                        distribution_column="popularity",
                        calibration_type=calibration_type
                    )
                elif calibration_column == 'inv_personalized':
                    if ratios_division[str(user)] <= ratio_mean :
                        if tradeoff == "VAR":
                            tradeoff_value = var_genre_all_users[user]
                        elif tradeoff == "GC":
                            tradeoff_value = cg_genre_all_users[user]
                        
                        re_ranked, _  = re_rank_list(
                        trainratings,
                        dataset.items,
                        user,
                        recomended_user[:100],
                        p_g_u_all_users=p_g_u_genre_all_users,
                        p_t_i_all_items=p_t_i_genre_all_items,
                        tradeoff=tradeoff_value,
                        N=10,
                        distribution_column="genres",
                        calibration_type=calibration_type
                    )
                    else:
                        if tradeoff == "VAR":
                            tradeoff_value = var_all_users[user]
                        elif tradeoff == "GC":
                            tradeoff_value = cg_all_users[user]

                        re_ranked, _  = re_rank_list(
                        trainratings,
                        dataset.items,
                        user,
                        recomended_user[:100],
                        p_g_u_all_users=p_g_u_all_users,
                        p_t_i_all_items=p_t_i_all_items,
                        tradeoff=tradeoff_value,
                        N=10,
                        distribution_column="popularity",
                        calibration_type=calibration_type
                    )
                else:

                    if tradeoff == "VAR":
                        tradeoff_value = var_all_users[user]
                        tradeoff_value_pop = var_all_users[user]
                    elif tradeoff == "GC":
                        tradeoff_value = cg_all_users[user]
                        tradeoff_value_pop = cg_all_users[user]

                    re_ranked, _  = re_rank_list(
                        trainratings,
                        dataset.items,
                        user,
                        recomended_user[:100],
                        p_g_u_all_users=p_g_u_all_users,
                        p_t_i_all_items=p_t_i_all_items,
                        tradeoff=tradeoff_value,
                        N=10,
                        distribution_column="popularity",
                        calibration_type=calibration_type
                    )
                
                recomend_results[calibration_column][tradeoff]["reranked"][user] = [
                    (item, index + 1)
                    for index, item in enumerate(re_ranked)
                ]
        
        return user, recomend_results
    except Exception as e:
            print("An error occurred in run_user:")
            print(e)


import time

def run_user3(
    train, dataset,
    calibration_column_list,
    model,
    trainratings,
    metadata,
    occurrences,
    vectorizer,
    profiles,
    valils, valmean,
    user


):
    recomend_results = {}

    start = time.time()

    aux = train[train["user"] == user]

    porofile = profiles[user]

    know_items_ids = (
        aux["item"].unique().tolist()
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
                    (pred.iid, pred.est)
                    for pred in predictions
                    if pred.uid == user
                ],
                key=lambda x: x[1],
                reverse=True,
            )

    recomended_user = aux_3

    for calibration_column in calibration_column_list:
        single, comparison_type = calibration_column


        if f"{single}_{comparison_type}" not in recomend_results:
            recomend_results[f"{single}_{comparison_type}"] = {}

        
        if single is True:
            tradeoff = 0.5
            if tradeoff not in recomend_results[f"{single}_{comparison_type}"]:
                recomend_results[f"{single}_{comparison_type}"][tradeoff] = {"reranked": {}}

            re_ranked, _  = re_rank_list2(
                trainratings,
                dataset.items,
                user,
                recomended_user[:30],
                N=10,
                user_profile_list=porofile,
                metadata=metadata,
                occurrences=occurrences,
                comparison_type=comparison_type,
                tradeoff=0.5,
                single=single,
                vectorizer=vectorizer, valils=valils, valmean=valmean
            )

            
            recomend_results[f"{single}_{comparison_type}"][tradeoff]["reranked"][user] = [
                (item, index + 1)
                for index, item in enumerate(re_ranked)
            ]
        else:

            for tradeoff in [0.25, 0.5, 0.75]:

                if tradeoff not in recomend_results[f"{single}_{comparison_type}"]:
                    recomend_results[f"{single}_{comparison_type}"][tradeoff] = {"reranked": {}}

                re_ranked, _  = re_rank_list2(
                    trainratings,
                    dataset.items,
                    user,
                    recomended_user[:30],
                    N=10,
                    user_profile_list=porofile,
                    metadata=metadata,
                    occurrences=occurrences,
                    comparison_type=comparison_type,
                    tradeoff=tradeoff,
                    single=single,
                    vectorizer=vectorizer, valils=valils, valmean=valmean
                )
                
                recomend_results[f"{single}_{comparison_type}"][tradeoff]["reranked"][user] = [
                    (item, index + 1)
                    for index, item in enumerate(re_ranked)
                ]
    print(f"Finished {start-time.time()}")
    return user, recomend_results






def run_user2(
    train, dataset,
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
    calibration_type,
    user


):
    start = time.time()
    recomend_results = {}

    

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
                    (pred.iid, pred.est)
                    for pred in predictions
                    if pred.uid == user
                ],
                key=lambda x: x[1],
                reverse=True,
            )

    recomended_user = aux_3

    


    for calibration_column in calibration_column_list:
        if calibration_column not in recomend_results:
            recomend_results[calibration_column] = {}

        if calibration_column == 'nothing':
            if 0 not in recomend_results[calibration_column]: recomend_results[calibration_column][0] = {'reranked': {}}
            recomend_results[calibration_column][0]['reranked'][user] =  recomended_user[:10]
        else:

            for tradeoff_first, tradeoff_first_value in zip(
                ["GC", 0.25, 0.5, 1], [cg_all_users[user], 0.25, 0.5, 1]
            ):
                if tradeoff_first not in recomend_results[calibration_column]:
                    recomend_results[calibration_column][tradeoff_first] = {}

                re_ranked, re_ranked_score = re_rank_list(
                        trainratings,
                        dataset.items,
                        user,
                        recomended_user[:200],
                        p_g_u_all_users=p_g_u_all_users,
                        p_t_i_all_items=p_t_i_all_items,
                        tradeoff=tradeoff_first_value,
                        N=100,
                        distribution_column="popularity",
                        calibration_type=calibration_type
                    )
                
                        
                for tradeoff, tradeoff_value in zip(
                    ["VAR", "GC", 0, 0.2, 0.4, 0.6, 0.8, 1], [var_genre_all_users[user], cg_genre_all_users[user], 0, 0.2, 0.4, 0.6, 0.8, 1]
                ):

                    if tradeoff not in recomend_results[calibration_column][tradeoff_first]:
                        recomend_results[calibration_column][tradeoff_first][tradeoff] = {"reranked": {}}

                    re_ranked = None
                        

                    re_ranked, re_ranked_score = re_rank_list(
                        trainratings,
                        dataset.items,
                        user,
                        re_ranked_score,
                        p_g_u_all_users=p_g_u_genre_all_users,
                        p_t_i_all_items=p_t_i_genre_all_items,
                        tradeoff=tradeoff_value,
                        N=10,
                        distribution_column="genres",
                        calibration_type=calibration_type
                    )
                    
                    recomend_results[calibration_column][tradeoff_first][tradeoff]["reranked"][user] = [
                        (item, index + 1)
                        for index, item in enumerate(re_ranked)
                    ]
    print(time.time() - start)
    return user, recomend_results


#######################3 PPTM



from scipy.stats import wasserstein_distance

def calculate_PPT(list_, popularity_bins, movie_budget, movies_bins):
    ppt = {k:0 for k in popularity_bins.keys()}
    sum_ = 0
    for item in list_:
        
        sum_ += np.log(movie_budget[item])
        b = movies_bins.get(item, 0)
        ppt[b] += np.log(movie_budget[item])
    
    
    for i in ppt.keys():
        ppt[i] = ppt[i]/sum_ if sum_ > 0 else 0
        
    return ppt


        
def seedset(list_items, user_ppt, popularity_bins, movie_budget, movies_bins, k=10):
    
    S = []
    S_ppt =  {k:0 for k in popularity_bins.keys()}
    K = list_items[:k]
    
    for ij, rating in K:
        b = movies_bins[ij]
        
        if user_ppt.get(b, 0) > S_ppt[b] + 1/k:
            S.append(ij)
            S_ppt[b] = S_ppt[b] + 1/k
            
    return S


def calculo(P, Q, popularity_bins, P_items_score, c=0.001):
    
    P_ = [P.get(bin_, 0) for bin_ in popularity_bins.keys()]
    Q_ = [Q.get(bin_, 0) for bin_ in popularity_bins.keys()]
    
    
    EMD = wasserstein_distance(P_, Q_)
    return P_items_score - c*EMD

def greedy_selection(list_items, S, B, user_profile_bins, itemscores, popularity_bins, movie_budget, movies_bins, k=10, c=0.001):
    R = S
    R_minus_S = []
    len_S = len(S)

    user_profile_score = 0
    for i in R:
        if isinstance(i, tuple):
            user_profile_score += itemscores[i[0]]
        else:
            user_profile_score += itemscores[i]

    
    for i in range(k-len_S):
        maxmmr = -np.inf
        maxitem = None
        for item, rating in B:
            if item in R:
                continue
            
            temp_list = R + [item]
            temp_score = user_profile_score + itemscores[item]

            objective = calculo(user_profile_bins, calculate_PPT(temp_list,popularity_bins, movie_budget, movies_bins), popularity_bins, temp_score, c=c)
            
            if objective > maxmmr:
                maxmmr = objective
                maxitem = item
        
        if maxitem is not None: 
            R.append(maxitem)
            user_profile_score += itemscores[item]
            R_minus_S.append(maxitem)
            B = [i for i in B if i[0] != maxitem]


    R_changed = True
    while R_changed is True:
        maxmmr = -np.inf
        maxitem = None
        temp_score = user_profile_score

        for (item, rating) in B:
            if item in R:
                continue
            for item2 in R_minus_S:
                
                temp_list1 = R + [item]
                if item2 in temp_list1: 
                    temp_list1.remove(item2)
                    temp_score = user_profile_score + itemscores[item] - itemscores[item2]
                else:

                    temp_score = user_profile_score + itemscores[item]
                
                objective = calculo(user_profile_bins, calculate_PPT(temp_list1,popularity_bins, movie_budget, movies_bins), popularity_bins, temp_score, c=c)
                
                if objective > maxmmr:
                    maxmmr = objective
                    maxitem = item, item2
        
        if maxitem is not None:
            temp_list1 = R + [maxitem[0]]
            if maxitem[1] in temp_list1: temp_list1.remove(maxitem[1])
            if calculo(
                user_profile_bins,
                calculate_PPT(temp_list1,popularity_bins, movie_budget, movies_bins), popularity_bins, temp_score, c=c
                ) > calculo(
                    user_profile_bins,
                    calculate_PPT(R,popularity_bins, movie_budget, movies_bins),
                    popularity_bins, temp_score, c=c
                ):
                
                
                R.append(maxitem[0])

                if maxitem[1] in R: 
                    R.remove(maxitem[1])
                    user_profile_score -= itemscores[maxitem[1]]

                user_profile_score += itemscores[maxitem[0]] 
                if maxitem[0] in B: 
                    B = [i for i in B if i[0] != maxitem[0]]
                
                B.append((maxitem[1], itemscores[maxitem[1]]))
                
                R_changed = True
            else:
                R_changed = False
        else:
            temp_list1 = R
            R_changed = False
        
        
            
            
        
    return R

import copy
def run_user_PPTM(
    train, dataset,
    calibration_column_list,
    model,
    trainratings,
    user_profile_bins,
    popularity_bins,
    movie_budget,
    movies_bins,
    user


):

    recomend_results = {}
    started = time.time()
    

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
                    (pred.iid, pred.est)
                    for pred in predictions
                    if pred.uid == user
                ],
                key=lambda x: x[1],
                reverse=True,
            )

    itemscores = {k:v for k, v in aux_3}

    recomended_user = copy.deepcopy(aux_3)


    S = seedset(
        recomended_user[:50],
        user_profile_bins.get(user, {}),
        popularity_bins,
        movie_budget,
        movies_bins,
        k=10
    )

    B = copy.deepcopy(aux_3)
    for i in S:
        if i in B: B.remove(i)


    recomend_results['PPTM'] = {}

    for c in [1, 0.5, 0.1, 0.01, 0.001]:
        if str(c) not in recomend_results['PPTM']:
            recomend_results['PPTM'][c] =  {"reranked": {}}
    
        R = greedy_selection(
            recomended_user[:50],
            S,
            B,
            user_profile_bins.get(user, {}),
            itemscores,
            popularity_bins,
            movie_budget,
            movies_bins,
            k=10,
            c=c
        )

        recomend_results['PPTM'][c]["reranked"][user] = [
            (item, index + 1) for index, item in enumerate(R)
        ]
    print(time.time() - started)
    return user, recomend_results



def run_user5(
    train, dataset,
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
    user_profile_bins,
    popularity_bins,
    movie_budget,
    movies_bins,
    movies_bins_profile,
    calibration_type,
    user


):
    start = time.time()
    recomend_results = {}

    

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
                    (pred.iid, pred.est)
                    for pred in predictions
                    if pred.uid == user
                ],
                key=lambda x: x[1],
                reverse=True,
            )

    recomended_user =aux_3
    


    for calibration_column in calibration_column_list:
        if calibration_column not in recomend_results:
            recomend_results[calibration_column] = {}

        if calibration_column == 'nothing':
            if 0 not in recomend_results[calibration_column]: recomend_results[calibration_column][0] = {'reranked': {}}
            recomend_results[calibration_column][0]['reranked'][user] =  recomended_user[:10]
        else:

            for tradeoff_first, tradeoff_first_value in zip(
                [0.25, 0.5, 1], [0.25, 0.5, 1]
            ):
                if tradeoff_first not in recomend_results[calibration_column]:
                    recomend_results[calibration_column][tradeoff_first] = {}


                re_ranked, re_ranked_score = re_rank_list_PPTM(
                        trainratings,
                        dataset.items,
                        user,
                        recomended_user[:70],
                        p_g_u_all_users=user_profile_bins,
                        p_t_i_all_items=movies_bins_profile,
                        tradeoff=tradeoff_first_value,
                        N=90,
                        distribution_column="budget",
                        popularity_bins=popularity_bins,
                        movie_budget=movie_budget,
                        movies_bins=movies_bins
                    )
                
                        
                for tradeoff, tradeoff_value in zip(
                    ["VAR", "GC", 0, 0.2, 0.4, 0.6, 0.8, 1], [var_genre_all_users[user], cg_genre_all_users[user], 0, 0.2, 0.4, 0.6, 0.8, 1]
                ):

                    if tradeoff not in recomend_results[calibration_column][tradeoff_first]:
                        recomend_results[calibration_column][tradeoff_first][tradeoff] = {"reranked": {}}

                    re_ranked = None
                        

                    re_ranked, re_ranked_score = re_rank_list(
                        trainratings,
                        dataset.items,
                        user,
                        re_ranked_score,
                        p_g_u_all_users=p_g_u_genre_all_users,
                        p_t_i_all_items=p_t_i_genre_all_items,
                        tradeoff=tradeoff_value,
                        N=10,
                        distribution_column="genres",
                        calibration_type=calibration_type
                    )
                    
                    recomend_results[calibration_column][tradeoff_first][tradeoff]["reranked"][user] = [
                        (item, index + 1)
                        for index, item in enumerate(re_ranked)
                    ]
    print(time.time() - start)
    return user, recomend_results
