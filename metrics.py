import numpy as np

class Metrics:
    @staticmethod
    def calculate_GAP_user_profile(group_list, train_data, popularity_items):
        if not group_list:
            return 0

        total_users_score = 0

        for user in group_list:
            if isinstance(user, list):
                user = user[0]

            user_interacted = train_data[train_data["user"] == user]
            user_score = sum(popularity_items.get(row["item"], 0) for _, row in user_interacted.iterrows())

            if len(user_interacted) != 0:
                total_users_score += user_score / len(user_interacted)

        if len(group_list) != 0:
            return total_users_score / len(group_list)
        return 0

    @staticmethod
    def calculate_GAP_recommendations(group_list, recommended_list, popularity_items):
        if not group_list:
            return 0

        total_users_score = 0

        for user in recommended_list:
            if user in group_list:
                user_score = sum(popularity_items.get(item, 0) for item, _ in recommended_list[user])

                if len(recommended_list[user]) != 0:
                    total_users_score += user_score / len(recommended_list[user])

        return total_users_score / len(group_list)

    @staticmethod
    def calculate_delta_GAP(group_list, recommended_list, train_data, popularity_items):
        if not group_list:
            return 0

        gap_p = Metrics.calculate_GAP_user_profile(group_list, train_data, popularity_items)
        gap_r = Metrics.calculate_GAP_recommendations(group_list, recommended_list, popularity_items)

        print("GAP Recommendations", gap_r)
        print("GAP User Profile", gap_p)
        
        if gap_p != 0:
            return gap_r / gap_p - 1
        return 0

    @staticmethod
    def calculate_target_item_distribution(movies_data, distribution_column = "genres", item_id = ""):
        try:
            item = movies_data[movies_data["item"] == item_id]

            if len(item) > 0:
                types = list(item[distribution_column])[0].split("|")
                distribution = {type_: 1 / len(types) for type_ in types}
                return item_id, distribution
            
        except Exception as e:
            print(f"Error calculating item distribution for Item ID {item_id}: {e}")
            
        return item_id, {}
    
    @staticmethod
    def calculate_target_user_distribution(
        dataset,
        movies_data,
        distribution_column = "genres",
        weight = "rating",
        target_all_items_distribution = None,
        user_id = ""
    ):
        genre_distribution = {}
        genre_weight = {}
        total_weight = 0

        user_interacted = dataset[dataset["user"] == user_id].drop_duplicates(subset=["item"])
        
        for _, row in user_interacted.iterrows():
            try:
                item_id = int(row["item"])
                
                if item_id in target_all_items_distribution:
                    item_distribution = target_all_items_distribution[item_id]
                else:
                    item_distribution = Metrics.calculate_target_item_distribution(
                        movies_data, distribution_column = distribution_column, item_id = item_id
                    )

                user_weight = row[weight]

                for genre in item_distribution.keys():
                    genre_score = genre_distribution.get(genre, 0.0)
                    genre_distribution[genre] = genre_score + (user_weight * item_distribution.get(genre, 0))

                    genre_weight_score = genre_weight.get(genre, 0.0)
                    genre_weight[genre] = genre_weight_score + user_weight
                    total_weight += user_weight
            except Exception as e:
                print(f"Error processing item for User ID {user_id}: {e}")
        
        if distribution_column == "popularity":
            genre_distribution = {genre: round(genre_score / total_weight, 3) for genre, genre_score in genre_distribution.items()}
        else:
            genre_distribution = {genre: round(genre_score / genre_weight[genre], 3) for genre, genre_score in genre_distribution.items()}

        return user_id, genre_distribution
    
    @staticmethod
    def get_user_KL_divergence(
        dataset,
        movies_data,
        user_id,
        recommended_items,
        alpha = 0.01,
        target_user_distribution = None,
        distribution_column = "genres",
        weight = "rating",
        target_all_items_distribution = None
    ):
        recommended_user_distribution = Metrics.calculate_recommended_user_distribution(
            recommended_items,
            movies_data,
            distribution_column = distribution_column,
            target_all_items_distribution = target_all_items_distribution,
        )

        if target_user_distribution is None:
            target_user_distribution = Metrics.calculate_target_user_distribution(
                dataset,
                movies_data,
                distribution_column = distribution_column,
                weight = weight,
                user_id = user_id
            )

        Ckl = 0

        for genre, p in target_user_distribution.items():
            q = recommended_user_distribution.get(genre, 0.0)
            til_q = (1 - alpha) * q + alpha * p

            if til_q == 0 or target_user_distribution.get(genre, 0) == 0:
                Ckl = Ckl
            else:
                Ckl += p * np.log2(p / til_q)
        return Ckl

    @staticmethod
    def calculate_recommended_user_distribution(
        recomended_items,
        movies_data,
        distribution_column = "genres",
        target_all_items_distribution = None,
    ):
        types_distribution = {}
        weigth = {}
        all_weight = 0

        for item, score in recomended_items:
            item_id = int(item)

            try:
                if item_id not in target_all_items_distribution:
                    item_distribution = Metrics.calculate_target_item_distribution(
                        movies_data,
                        distribution_column = distribution_column,
                        item_id = item_id
                    )
                else:
                    item_distribution = target_all_items_distribution[item_id]

                if item_distribution:
                    w_ri = score

                    for type_ in item_distribution.keys():
                        types_score = types_distribution.get(type_, 0.0)
                        types_distribution[type_] = types_score + (
                            w_ri * item_distribution.get(type_, 0)
                        )

                        weigth_rating = weigth.get(type_, 0.0)
                        weigth[type_] = weigth_rating + w_ri
                        all_weight += w_ri

            except Exception as e:
                pass
                #print(f"Error processing item: {e}")

        for type_, type_score in types_distribution.items():
            if distribution_column == "popularity":
                normalized_weight = type_score / all_weight
            else:
                normalized_weight = type_score / weigth[type_]
            normed_type_score = round(normalized_weight, 3)
            types_distribution[type_] = normed_type_score

        return types_distribution

    def get_mean_rank_miscalibration(
        movies,
        trainratings,
        recomendation_list_per_user,
        distribution_column = "genres",
        target_all_users_distribution = None,
        target_all_items_distribution = None
    ):
        MRMC = 0
        for user_id in recomendation_list_per_user.keys():
            MRMC += Metrics.get_user_rank_miscalibration(
                movies,
                trainratings,
                user_id,
                recomendation_list_per_user[user_id],
                distribution_column = distribution_column,
                target_all_users_distribution = target_all_users_distribution,
                target_all_items_distribution = target_all_items_distribution,
            )

        if len(recomendation_list_per_user.keys()) == 0: return 0
        return MRMC / len(recomendation_list_per_user.keys())

    def get_miscalibration(
        dataset,
        movies,
        void,
        user,
        recomendation,
        fairness_function = "KL",
        target_user_distribution = None,
        distribution_column = "genres",
        target_all_users_distribution = None,
        target_all_items_distribution = None
    ):
        if void != 0:
            if fairness_function.lower() == "kl":
                kl = Metrics.get_user_KL_divergence(
                    dataset,
                    movies,
                    user,
                    recomendation,
                    target_user_distribution = target_user_distribution,
                    distribution_column = distribution_column,
                    target_all_items_distribution = target_all_items_distribution,
                )
                return kl / void

        return 0

    def get_user_rank_miscalibration(
        movies,
        trainratings,
        user_id,
        recommended_items,
        fairness_function = "KL",
        distribution_column = "genres",
        target_all_users_distribution = None,
        target_all_items_distribution = None
    ):
        RMC = 0
        N = len(recommended_items)

        if user_id not in target_all_users_distribution:
            target_user_distribution = Metrics.calculate_target_user_distribution(
                trainratings,
                movies,
                weight = "rating",
                distribution_column = distribution_column,
                user_id = user_id
            )
        else:
            target_user_distribution = target_all_users_distribution[user_id]

        if fairness_function.lower() == "kl":
            void = Metrics.get_user_KL_divergence(
                trainratings,
                movies,
                user_id,
                [],
                target_user_distribution = target_user_distribution,
                distribution_column = distribution_column,
                target_all_items_distribution = target_all_items_distribution
            )

        for i in range(1, N+1):
            partial_recomendation = recommended_items[:i]
            RMC += Metrics.get_miscalibration(
                trainratings,
                movies,
                void,
                user_id,
                partial_recomendation,
                fairness_function,
                target_user_distribution = target_user_distribution,
                distribution_column = distribution_column,
                target_all_items_distribution = target_all_items_distribution,
                target_all_users_distribution = target_all_users_distribution
            )
        if N == 0: return 0 
        return RMC / N

    @staticmethod
    def get_user_calibration_error(
        movies,
        recomended_items,
        target_user_distribution,
        distribution_column = "genres",
        target_all_items_distribution = None,
    ):
        recommended_user_distribution = Metrics.calculate_recommended_user_distribution(
            recomended_items,
            movies,
            distribution_column = distribution_column,
            target_all_items_distribution = target_all_items_distribution,
        )

        error = [
            abs(target_user_distribution.get(genre, 0) - recommended_user_distribution.get(genre, 0))
            for genre in recommended_user_distribution.keys()
        ]

        if len(target_user_distribution.keys()) != 0:
            return sum(error) / len(target_user_distribution.keys())
        return 0

    @staticmethod
    def get_user_average_calibration_error(
        movies,
        trainratings,
        user_id,
        recommended_items,
        distribution_column = "genres",
        target_all_users_distribution = None,
        target_all_items_distribution = None,
    ):
        ACE = 0
        N = len(recommended_items)

        if target_all_users_distribution is None:
            target_user_distribution = Metrics.calculate_target_user_distribution(
                trainratings,
                movies,
                weight = "rating",
                distribution_column = distribution_column,
                user_id = user_id
            )
        else:
            if user_id not in target_all_users_distribution:
                target_user_distribution = Metrics.calculate_target_user_distribution(
                trainratings,
                movies,
                weight = "rating",
                distribution_column = distribution_column,
                user_id = user_id
            )
            else:
                target_user_distribution = target_all_users_distribution[user_id]

        for i in range(1, N+1):
            partial_recomendation = recommended_items[:i]
            ACE += Metrics.get_user_calibration_error(
                movies,
                partial_recomendation,
                target_user_distribution = target_user_distribution,
                distribution_column = distribution_column,
                target_all_items_distribution = target_all_items_distribution,
            )

        if N == 0: return 0
        return ACE / N

    @staticmethod
    def get_mean_average_calibration_error(
        movies,
        trainratings,
        recomendation_list_per_user,
        distribution_column = "genres",
        target_all_users_distribution = None,
        target_all_items_distribution = None,
    ):

        MACE = 0

        for user_id in recomendation_list_per_user.keys():
            MACE += Metrics.get_user_average_calibration_error(
                movies,
                trainratings,
                user_id,
                recomendation_list_per_user[user_id],
                distribution_column = distribution_column,
                target_all_users_distribution = target_all_users_distribution,
                target_all_items_distribution = target_all_items_distribution,
            )

        if len(recomendation_list_per_user.keys()) == 0: return 0
        return MACE / len(recomendation_list_per_user.keys())

    @staticmethod
    def long_tail_coverage(recomendations_per_user, all_items, isPairwise = False):
        m_t_items = all_items[all_items["popularity"].isin(["M", "T"])]

        subset_m_t = list(m_t_items["item"].unique())
        union_recommended_m_t = []

        for user in recomendations_per_user:
            for item, _ in recomendations_per_user[user]:
                if (isPairwise):
                    if np.any(subset_m_t == item):
                        union_recommended_m_t.append(item)
                else:
                    if item in subset_m_t:
                        union_recommended_m_t.append(item)

        return len(set(union_recommended_m_t)) / len(m_t_items)

    @staticmethod
    def aggregate_diversity(recomendations_per_user, all_items):
        set_recommended_items = []

        for user in recomendations_per_user:
            set_recommended_items = set_recommended_items + [
                i[0] for i in recomendations_per_user[user]
            ]

        set_recommended_items = set(set_recommended_items)
        
        return len(set_recommended_items) / len(all_items)

    @staticmethod
    def prec(recomendations_per_user, testratings, n=10):
        n_relev = 0
        n_ = 0
        for user in recomendations_per_user:
            n_relevant = 0
            aux = testratings[testratings["user"] == user]
            y_pred = sorted(recomendations_per_user[user], key=lambda x: x[1])[:n+1]

            for item, _ in y_pred:
                n_ += 1
                if item in list(aux["item"]):
                        n_relevant += 1

            n_relev += float(n_relevant)
        if n_ != 0:
            return n_relev / n_
        return 0

    @staticmethod
    def tradeoff_genre_count(
        all_genres,
        trainratings_data,
        movies_data,
        user,
        distribution_column = "genres",
        target_all_users_distribution = None,
    ):
        if user not in target_all_users_distribution:
            target_user_distribution = Metrics.calculate_target_user_distribution(
                trainratings_data,
                movies_data,
                distribution_column = distribution_column,
                user_id = user
            )
        else:
            target_user_distribution = target_all_users_distribution[user]
        return len(target_user_distribution) / len(all_genres)

    def tradeoff_variance(
        self,
        all_genres,
        trainratings_data,
        movies_data,
        user,
        distribution_column = "genres",
        target_all_users_distribution = None,
    ):
        if target_all_users_distribution is not None:
            if user not in target_all_users_distribution:
                target_user_distribution = Metrics.calculate_target_user_distribution(
                    trainratings_data,
                    movies_data,
                    distribution_column = distribution_column,
                    user_id = user
                )
            else:
                target_user_distribution = target_all_users_distribution[user]

            if isinstance(all_genres, list):
                mean_genre = 0
                for genre in target_user_distribution:
                    mean_genre += target_user_distribution[genre]
                mean_genre /= len(all_genres)

                variance = 0

                for genre in all_genres:
                    variance += (target_user_distribution.get(genre, 0) - mean_genre) ** 2

                return 1 - (variance / len(all_genres))

    @staticmethod
    def item_is_relevant(testratings, user_id, item_id):
        aux = testratings[testratings["user"] == user_id]

        if item_id in list(aux["item"]):
            return True
        return False

    @staticmethod
    def user_relevant_items(testratings, user_id):
        aux = testratings[testratings["user"] == user_id]
        return len(aux["item"].unique())

    @staticmethod
    def ap_at(testratings, user_id, recommendation_list, N=10):
        AP = 0

        for k in range(N):
            AP += Metrics.prec({user_id: recommendation_list}, testratings=testratings, n=k)
        if N == 0: return 0
        return AP / N


    @staticmethod
    def map_at(testratings, recomendations_per_user, N=10):
        MAP = 0
        for user_id in recomendations_per_user:
            AP = Metrics.ap_at(testratings = testratings, user_id = user_id, recommendation_list = recomendations_per_user[user_id], N = N)
                        
            MAP += AP
        if len(recomendations_per_user.keys()) != 0:
            return MAP / len(recomendations_per_user.keys())
        return 0

    @staticmethod
    def mrr(testratings, recomendations_per_user):
        MRR = 0

        for user_id in recomendations_per_user:
            user_find = False
            user_mrr = 0
            for item_id, index in recomendations_per_user[user_id]:
                if user_find == False:
                    if Metrics.item_is_relevant(testratings, user_id = user_id, item_id = item_id):
                        MRR += (1/index)
                        user_mrr += (1/index)
                        user_find = True
                        break
                else:
                    break
        if len(recomendations_per_user.keys()) != 0:    
            return MRR / len(recomendations_per_user.keys())
        return 0
