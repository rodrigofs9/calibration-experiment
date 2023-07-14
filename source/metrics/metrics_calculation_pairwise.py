from source.metrics.metrics import Metrics


def calculate_tradeoff_metrics(
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
    BB_group,
    N_group,
    D_group,
    popularity_items,
    popularity_bins=None,
    movie_budget=None,
    movies_bins=None,
    user_profile_bins=None,
    movies_bins_profile=None,
    isPairwise=False,
    tradeoff=None
):
    metrics_df = [model_name, ind, calibration_column, tradeoff]
    recomended_list = {user_id_: datas[calibration_column][tradeoff]["reranked"][user_id_] for user_id_, datas in exp_results}

    #print(recomended_list)

    # recomended_list = recomend_results[model_name][calibration_column][tradeoff]["reranked"]
    result = Metrics.get_mean_average_calibration_error(
        dataset,
        trainratings,
        recomended_list,
        p_t_u_all_users=p_g_u_genre_all_users,
        p_t_i_all_items=p_t_i_genre_all_items,
    )
    metrics_df.append(result)
    print("MACE_GENRES", result)

    result = Metrics.get_mean_average_calibration_error(
        dataset,
        trainratings,
        recomended_list,
        distribution_column="popularity",
        p_t_u_all_users=p_g_u_all_users,
        p_t_i_all_items=p_t_i_all_items,
    )
    print("MACE_POP", result)
    metrics_df.append(result)

    result = Metrics.long_tail_coverage(recomended_list, dataset, isPairwise)
    print("LTC", result)
    metrics_df.append(result)

    result = Metrics.get_mean_rank_miscalibration(
        dataset,
        trainratings,
        recomended_list,
        distribution_column="genres",
        p_t_u_all_users=p_g_u_genre_all_users,
        p_t_i_all_items=p_t_i_genre_all_items,
    )
    print("MRMC_GENRES", result)
    metrics_df.append(result)

    result = Metrics.get_mean_rank_miscalibration(
        dataset,
        trainratings,
        recomended_list,
        distribution_column="popularity",
        p_t_u_all_users=p_g_u_all_users,
        p_t_i_all_items=p_t_i_all_items,
    )
    print("MRMC_POP", result)
    metrics_df.append(result)

    # ## MRMC pop revenue

    # result = Metrics.get_mean_rank_miscalibration(
    #     dataset.items,
    #     trainratings,
    #     recomended_list,
    #     distribution_column="budget",
    #     p_t_u_all_users=user_profile_bins,
    #     p_t_i_all_items=movies_bins_profile,
    #     popularity_bins=popularity_bins,
    #     movie_budget=movie_budget,
    #     movies_bins=movies_bins,
    # )
    # print("MRMC_POP_BUDGET", result)
    # metrics_df.append(result)

    # ## END

    result = Metrics.aggregate_diversity(
        recomended_list, dataset, isPairwise
    )
    print("AGG-DIV", result)
    metrics_df.append(result)

    result = Metrics.prec(recomended_list, test, 10, isPairwise)
    print("Prec@10", result)
    metrics_df.append(result)

    result = Metrics.map_at(test, recomended_list, isPairwise=isPairwise, N=10)
    print("MAP@10", result)
    metrics_df.append(result)

    result = Metrics.mrr(test, recomended_list, isPairwise)
    print("MRR", result)
    metrics_df.append(result)

    result = Metrics.delta_GAP(BB_group, recomended_list, trainratings, popularity_items, isPairwise)
    print("GAPBB", result)
    metrics_df.append(result)

    result = Metrics.delta_GAP(N_group, recomended_list, trainratings, popularity_items, isPairwise)
    print("GAPN", result)
    metrics_df.append(result)

    result = Metrics.delta_GAP(D_group, recomended_list, trainratings, popularity_items, isPairwise)
    print("GAPD", result)
    metrics_df.append(result)

    print(f"\n\n-----------------------------------------------------------------------------")

    return metrics_df

def calculate_tradeoff_metrics2(
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
    BB_group,
    N_group,
    D_group,
    popularity_items,
    popularity_bins=None,
    movie_budget=None,
    movies_bins=None,
    user_profile_bins=None,
    movies_bins_profile=None,
    isPairwise=False,
    tradeoff=None
):

    tradeoff_first, tradeoff = tradeoff

    metrics_df = [model_name, ind, calibration_column, tradeoff_first, tradeoff]
    recomended_list = {user_id_: datas[calibration_column][tradeoff_first][tradeoff]["reranked"][user_id_] for user_id_, datas in exp_results}

    # recomended_list = recomend_results[model_name][calibration_column][tradeoff]["reranked"]
    result = Metrics.get_mean_average_calibration_error(
        dataset.items,
        trainratings,
        recomended_list,
        p_t_u_all_users=p_g_u_genre_all_users,
        p_t_i_all_items=p_t_i_genre_all_items,
    )
    metrics_df.append(result)
    print("MACE_GENRES", result)

    result = Metrics.get_mean_average_calibration_error(
        dataset.items,
        trainratings,
        recomended_list,
        distribution_column="popularity",
        p_t_u_all_users=p_g_u_all_users,
        p_t_i_all_items=p_t_i_all_items,
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
        distribution_column="genres",
        p_t_u_all_users=p_g_u_genre_all_users,
        p_t_i_all_items=p_t_i_genre_all_items,
    )
    print("MRMC_GENRES", result)
    metrics_df.append(result)

    result = Metrics.get_mean_rank_miscalibration(
        dataset.items,
        trainratings,
        recomended_list,
        distribution_column="popularity",
        p_t_u_all_users=p_g_u_all_users,
        p_t_i_all_items=p_t_i_all_items,
    )
    print("MRMC_POP", result)
    metrics_df.append(result)

    ## MRMC pop revenue

    result = Metrics.get_mean_rank_miscalibration(
        dataset.items,
        trainratings,
        recomended_list,
        distribution_column="budget",
        p_t_u_all_users=user_profile_bins,
        p_t_i_all_items=movies_bins_profile,
        popularity_bins=popularity_bins,
        movie_budget=movie_budget,
        movies_bins=movies_bins
    )
    print("MRMC_POP_BUDGET", result)
    metrics_df.append(result)

    ## END

    result = Metrics.aggregate_diversity(
        recomended_list, dataset.items
    )

    print("AGG-DIV", result)
    metrics_df.append(result)

    result = Metrics.prec(recomended_list, test, isPairwise)
    print("Prec@10", result)
    metrics_df.append(result)

    result = Metrics.map_at(test, recomended_list, N=10)
    print("MAP@10", result)
    metrics_df.append(result)

    result = Metrics.mrr(test, recomended_list)
    print("MRR", result)
    metrics_df.append(result)

    result = Metrics.delta_GAP(BB_group, recomended_list, trainratings, popularity_items, isPairwise)
    print("GAPBB", result)
    metrics_df.append(result)

    result = Metrics.delta_GAP(N_group, recomended_list, trainratings, popularity_items, isPairwise)
    print("GAPN", result)
    metrics_df.append(result)

    result = Metrics.delta_GAP(D_group, recomended_list, trainratings, popularity_items, isPairwise)
    print("GAPD", result)
    metrics_df.append(result)


    print(f"\n\n-----------------------------------------------------------------------------")

    return metrics_df