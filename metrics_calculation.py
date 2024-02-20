from metrics import Metrics

def calculate_tradeoff_metrics(
    model_name, fold,
    calibration_column,
    target_all_users_genres_distribution,
    target_all_items_genres_distribution,
    dataset,
    trainratings,
    target_all_users_distribution,
    target_all_items_distribution,
    test,
    exp_results,
    BB_group,
    N_group,
    D_group,
    popularity_items,
    tradeoff = None
):
    metrics_df = [model_name, fold, calibration_column, tradeoff]
    
    filtered_exp_results = [item for item in exp_results if item is not None]
    recomended_list = {}
    try:
        recomended_list = {user_id_: datas[calibration_column][tradeoff]["reranked"][user_id_] for user_id_, datas in filtered_exp_results}  
    except Exception as e:
        print(f"An exception occurred: {e}")

    # recomended_list = recomend_results[model_name][calibration_column][tradeoff]["reranked"]

    result = Metrics.get_mean_average_calibration_error(
        dataset if (model_name == "pairwise") else dataset.items,
        trainratings,
        recomended_list,
        target_all_users_distribution = target_all_users_genres_distribution,
        target_all_items_distribution = target_all_items_genres_distribution,
    )
    metrics_df.append(result)
    print("MACE_GENRES", result)

    result = Metrics.get_mean_average_calibration_error(
        dataset if (model_name == "pairwise") else dataset.items,
        trainratings,
        recomended_list,
        distribution_column = "popularity",
        target_all_users_distribution = target_all_users_distribution,
        target_all_items_distribution = target_all_items_distribution,
    )
    print("MACE_POP", result)
    metrics_df.append(result)

    result = Metrics.long_tail_coverage(
        recomended_list, 
        dataset if (model_name == "pairwise") else dataset.items, 
        True if (model_name == "pairwise") else False)
    print("LTC", result)
    metrics_df.append(result)

    result = Metrics.get_mean_rank_miscalibration(
        dataset if (model_name == "pairwise") else dataset.items,
        trainratings,
        recomended_list,
        distribution_column = "genres",
        target_all_users_distribution = target_all_users_genres_distribution,
        target_all_items_distribution = target_all_items_genres_distribution,
    )
    print("MRMC_GENRES", result)
    metrics_df.append(result)

    result = Metrics.get_mean_rank_miscalibration(
        dataset if (model_name == "pairwise") else dataset.items,
        trainratings,
        recomended_list,
        distribution_column = "popularity",
        target_all_users_distribution = target_all_users_distribution,
        target_all_items_distribution = target_all_items_distribution,
    )
    print("MRMC_POP", result)
    metrics_df.append(result)

    result = Metrics.aggregate_diversity(recomended_list, dataset if (model_name == "pairwise") else dataset.items)
    print("AGG-DIV", result)
    metrics_df.append(result)

    result = Metrics.prec(recomended_list, test, 10)
    print("Prec@10", result)
    metrics_df.append(result)

    result = Metrics.map_at(test, recomended_list, N = 10)
    print("MAP@10", result)
    metrics_df.append(result)

    result = Metrics.mrr(test, recomended_list)
    print("MRR", result)
    metrics_df.append(result)

    result = Metrics.calculate_delta_GAP(BB_group, recomended_list, trainratings, popularity_items)
    print("GAPBB", result)
    metrics_df.append(result)

    result = Metrics.calculate_delta_GAP(N_group, recomended_list, trainratings, popularity_items)
    print("GAPN", result)
    metrics_df.append(result)

    result = Metrics.calculate_delta_GAP(D_group, recomended_list, trainratings, popularity_items)
    print("GAPD", result)
    metrics_df.append(result)

    print(f"\n\n-----------------------------------------------------------------------------")

    return metrics_df

def calculate_tradeoff_metrics2(
    model_name, fold,
    calibration_column,
    target_all_users_genres_distribution,
    target_all_items_genres_distribution,
    dataset,
    trainratings,
    target_all_users_distribution,
    target_all_items_distribution,
    test,
    exp_results,
    BB_group,
    N_group,
    D_group,
    popularity_items,
    tradeoff = None
):

    tradeoff_first, tradeoff = tradeoff

    metrics_df = [model_name, fold, calibration_column, tradeoff_first, tradeoff]
    recomended_list = {user_id_: datas[calibration_column][tradeoff_first][tradeoff]["reranked"][user_id_] for user_id_, datas in exp_results}

    # recomended_list = recomend_results[model_name][calibration_column][tradeoff]["reranked"]
    result = Metrics.get_mean_average_calibration_error(
        dataset.items,
        trainratings,
        recomended_list,
        target_all_users_distribution = target_all_users_genres_distribution,
        target_all_items_distribution = target_all_items_genres_distribution,
    )
    metrics_df.append(result)
    print("MACE_GENRES", result)

    result = Metrics.get_mean_average_calibration_error(
        dataset.items,
        trainratings,
        recomended_list,
        distribution_column = "popularity",
        target_all_users_distribution = target_all_users_distribution,
        target_all_items_distribution = target_all_items_distribution,
    )
    print("MACE_POP", result)
    metrics_df.append(result)

    result = Metrics.long_tail_coverage(recomended_list, dataset.items)
    print("LTC", result)
    metrics_df.append(result)

    result = Metrics.get_mean_rank_miscalibration(
        dataset.items,
        trainratings,
        recomended_list,
        distribution_column = "genres",
        target_all_users_distribution = target_all_users_genres_distribution,
        target_all_items_distribution = target_all_items_genres_distribution,
    )
    print("MRMC_GENRES", result)
    metrics_df.append(result)

    result = Metrics.get_mean_rank_miscalibration(
        dataset.items,
        trainratings,
        recomended_list,
        distribution_column="popularity",
        target_all_users_distribution = target_all_users_distribution,
        target_all_items_distribution = target_all_items_distribution,
    )
    print("MRMC_POP", result)
    metrics_df.append(result)

    result = Metrics.aggregate_diversity(recomended_list, dataset.items)

    print("AGG-DIV", result)
    metrics_df.append(result)

    result = Metrics.prec(recomended_list, test)
    print("Prec@10", result)
    metrics_df.append(result)

    result = Metrics.map_at(test, recomended_list, N = 10)
    print("MAP@10", result)
    metrics_df.append(result)

    result = Metrics.mrr(test, recomended_list)
    print("MRR", result)
    metrics_df.append(result)

    result = Metrics.calculate_delta_GAP(BB_group, recomended_list, trainratings, popularity_items)
    print("GAPBB", result)
    metrics_df.append(result)

    result = Metrics.calculate_delta_GAP(N_group, recomended_list, trainratings, popularity_items)
    print("GAPN", result)
    metrics_df.append(result)

    result = Metrics.calculate_delta_GAP(D_group, recomended_list, trainratings, popularity_items)
    print("GAPD", result)
    metrics_df.append(result)

    print(f"\n\n-----------------------------------------------------------------------------")

    return metrics_df