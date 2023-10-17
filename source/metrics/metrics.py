import multiprocessing
from os import stat
import numpy as np
import time

def calculate_PPT(list_, popularity_bins, movie_budget, movies_bins):
    ppt = {k:0 for k in popularity_bins.keys()}
    sum_ = 0
    for item, rating in list_:
        
        sum_ += np.log(movie_budget[item])
        b = movies_bins.get(item, 0)
        ppt[b] += np.log(movie_budget[item])
    
    
    for i in ppt.keys():
        ppt[i] = ppt[i]/sum_ if sum_ > 0 else 0
        
    return ppt

class Metrics:
    @staticmethod
    def get_p_t_i_distribution_mp(
         movies_data, distribution_column="genres", item_id="", movies_budget=None, movies_revenues=None
    ):  
        if distribution_column == "budget":
            return item_id, {}
        elif distribution_column == 'revenue':
            return item_id, {}
        else:
            item = movies_data[movies_data["item"] == item_id]
            if len(item) > 0:
                types = list(item[distribution_column])[0].split("|")
                distribution = {type_: 1 / len(types) for type_ in types}
                return item_id, distribution
            return item_id, {}
    
    @staticmethod
    def get_p_t_i_distribution(
        item_id, movies_data, distribution_column="genres"
    ):
        item = movies_data[movies_data["item"] == item_id]
        try:
            if len(item) > 0:
                types = list(item[distribution_column])[0].split("|")
                distribution = {type_: 1 / len(types) for type_ in types}
                return distribution
        except Exception as e:
            print("error", e)
            time.sleep(300)
        return None


    @staticmethod
    def get_p_t_u_distribution_mp(
        dataset,
        movies_data,
        distribution_column="genres",
        weight="rating",
        p_t_i_all_items=None,
        user_id=''
        
    ):
        types_distribution = {}
        weigth = {}
        all_weight = 0

        user_interacted = dataset[dataset["user"] == user_id]
        user_interacted = user_interacted.drop_duplicates(subset=['item'])
        for _, row in user_interacted.iterrows():
            try:
                if int(row["item"]) in p_t_i_all_items:
                    p_t_i = p_t_i_all_items[int(row["item"])]
                else:
                    p_t_i = Metrics.get_p_t_i_distribution(int(row["item"]), movies_data, distribution_column=distribution_column)

                w_ui = row[weight]


                for genre in p_t_i.keys():

                    type_score = types_distribution.get(genre, 0.0)
                    types_distribution[genre] = type_score + (
                        w_ui * p_t_i.get(genre, 0)
                    )

                    weigth_rating = weigth.get(genre, 0.0)
                    weigth[genre] = weigth_rating + w_ui
                    all_weight += w_ui
            except:
                ...
        

        if distribution_column == "popularity":
            types_distribution = {type_:round(type_score / all_weight, 3) for type_,type_score in types_distribution.items()}
        else:
            types_distribution = {type_:round(type_score / weigth[type_], 3) for type_,type_score in types_distribution.items()}
    
        return user_id, types_distribution

    @staticmethod
    def GAP_p(group_list, train_data, popularity_items):
        if len(group_list) == 0: return 0
        sum_users = 0
        for user in group_list:
            sum_items = 0
            if isinstance(user, list):
                user = user[0]
            user_interacted = train_data[train_data["user"] == user]
            for _, row in user_interacted.iterrows():
                sum_items += popularity_items.get(row['item'], 0)
            
            if len(user_interacted) != 0: sum_users += sum_items/len(user_interacted)

        if len(group_list) != 0:
            return sum_users/len(group_list)
        return 0

    @staticmethod
    def GAP_r(group_list, recommended_list, popularity_items, isPairwise=False):
        if len(group_list) == 0: return 0
        sum_users = 0
        for user in recommended_list:
            if user in group_list:
                sum_items = 0
                for item, _ in recommended_list[user]:
                    if(isPairwise):
                        sum_items += popularity_items.get(tuple(item), 0)
                    else:
                        sum_items += popularity_items.get(item, 0)
                
                if len(recommended_list[user]) != 0: sum_users += sum_items/len(recommended_list[user])

        return sum_users/len(group_list)
    
    @staticmethod
    def delta_GAP(group_list, recommended_list, train_data, popularity_items, isPairwise):
        if len(group_list) == 0: return 0
        gap_p = Metrics.GAP_p(group_list, train_data, popularity_items)
        gap_r = Metrics.GAP_r(group_list, recommended_list, popularity_items, isPairwise)
        print("GAP R", gap_r)
        print("GAP P", gap_p)
        print(gap_r/gap_p)
        return gap_r/gap_p - 1


    @staticmethod
    def get_p_t_u_distribution(
        dataset,
        movies_data,
        user_id,
        distribution_column="genres",
        weight="rating",
        p_t_i_all_items=None,
    ):
        types_distribution = {}
        weigth = {}
        all_weight = 0

        user_interacted = dataset[dataset["user"] == user_id]
        user_interacted = user_interacted.drop_duplicates(subset=['item'])
        for _, row in user_interacted.iterrows():
            try:
                if int(row["item"]) in p_t_i_all_items:
                    p_t_i = p_t_i_all_items[int(row["item"])]
                else:
                    p_t_i = Metrics.get_p_t_i_distribution(int(row["item"]), movies_data, distribution_column=distribution_column)

                w_ui = row[weight]


                for genre in p_t_i.keys():

                    type_score = types_distribution.get(genre, 0.0)
                    types_distribution[genre] = type_score + (
                        w_ui * p_t_i.get(genre, 0)
                    )

                    weigth_rating = weigth.get(genre, 0.0)
                    weigth[genre] = weigth_rating + w_ui
                    all_weight += w_ui
            except:
                ...
        

        if distribution_column == "popularity":
            types_distribution = {type_:round(type_score / all_weight, 3) for type_,type_score in types_distribution.items()}
        else:
            types_distribution = {type_:round(type_score / weigth[type_], 3) for type_,type_score in types_distribution.items()}
       
        return types_distribution

    @staticmethod
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

    @staticmethod
    def get_user_KL_divergence(
        dataset,
        movies_data,
        user_id,
        recomended_items,
        alpha=0.01,
        p_g_u=None,
        distribution_column="genres",
        weight="rating",
        p_t_u_all_users=None,
        p_t_i_all_items=None,
        popularity_bins=None,
        movie_budget=None,
        movies_bins=None,
    ):
        q_g_u = Metrics.get_q_t_u_distribution(
            recomended_items,
            movies_data,
            popularity_bins=popularity_bins,
            movie_budget=movie_budget,
            movies_bins=movies_bins,
            distribution_column=distribution_column,
            p_t_i_all_items=p_t_i_all_items,
        )

        if p_g_u is None:
            p_g_u = Metrics.get_p_t_u_distribution(
                dataset,
                movies_data,
                user_id,
                distribution_column=distribution_column,
                weight=weight,
            )

        Ckl = 0

        for genre, p in p_g_u.items():
            q = q_g_u.get(genre, 0.0)
            til_q = (1 - alpha) * q + alpha * p

            if til_q == 0 or p_g_u.get(genre, 0) == 0:
                Ckl = Ckl
            else:
                Ckl += p * np.log2(p / til_q)
        return Ckl

    @staticmethod
    def get_q_t_u_distribution(
        recomended_items,
        movies_data,
        popularity_bins=None,
        movie_budget=None,
        movies_bins=None,
        distribution_column="genres",
        p_t_i_all_items=None,
    ):

        if distribution_column == 'revenue':
            return calculate_PPT(recomended_items, popularity_bins, movie_budget, movies_bins)
        elif distribution_column == 'budget':
            return calculate_PPT(recomended_items, popularity_bins, movie_budget, movies_bins)
        else:
            types_distribution = {}
            weigth = {}
            all_weight = 0
            for item, score in recomended_items:
                if int(item) not in p_t_i_all_items:
                    p_g_i = Metrics.get_p_t_i_distribution(
                        int(item),
                        movies_data,
                        distribution_column=distribution_column,
                    )
                else:
                    p_g_i = p_t_i_all_items[int(item)]
                if p_g_i:

                    w_ri = score

                    for type_ in p_g_i.keys():

                        types_score = types_distribution.get(type_, 0.0)
                        types_distribution[type_] = types_score + (
                            w_ri * p_g_i.get(type_, 0)
                        )

                        weigth_rating = weigth.get(type_, 0.0)
                        weigth[type_] = weigth_rating + w_ri
                        all_weight += w_ri

            for type_, type_score in types_distribution.items():
                if distribution_column == "popularity":
                    w = type_score / all_weight
                else:
                    w = type_score / weigth[type_]
                normed_type_score = round(w, 3)
                types_distribution[type_] = normed_type_score

            return types_distribution

    def get_mean_rank_miscalibration(
        movies,
        trainratings,
        recomendation_list_per_user,
        distribution_column="genres",
        p_t_u_all_users=None,
        p_t_i_all_items=None,
        popularity_bins=None,
        movie_budget=None,
        movies_bins=None,
    ):
        MRMC = 0
        for user_id in recomendation_list_per_user.keys():
            MRMC += Metrics.get_user_rank_miscalibration(
                movies,
                trainratings,
                user_id,
                recomendation_list_per_user[user_id],
                distribution_column=distribution_column,
                p_t_u_all_users=p_t_u_all_users,
                p_t_i_all_items=p_t_i_all_items,
                popularity_bins=popularity_bins,
                movie_budget=movie_budget,
                movies_bins=movies_bins,
            )

        if len(recomendation_list_per_user.keys()) == 0: return 0
        return MRMC / len(recomendation_list_per_user.keys())

    def get_miscalibration(
        dataset,
        movies,
        void,
        user,
        recomendation,
        fairness_function="KL",
        p_g_u=None,
        distribution_column="genres",
        p_t_u_all_users=None,
        p_t_i_all_items=None,
        popularity_bins=None,
        movie_budget=None,
        movies_bins=None,
    ):
        if void != 0:
            if fairness_function.lower() == "kl":
                kl = Metrics.get_user_KL_divergence(
                    dataset,
                    movies,
                    user,
                    recomendation,
                    p_g_u=p_g_u,
                    distribution_column=distribution_column,
                    p_t_u_all_users=p_t_u_all_users,
                    p_t_i_all_items=p_t_i_all_items,
                    popularity_bins=popularity_bins,
                movie_budget=movie_budget,
                movies_bins=movies_bins,
                )
                return kl / void

        return 0

    def get_user_rank_miscalibration(
        movies,
        trainratings,
        user_id,
        recomended_items,
        fairness_function="KL",
        distribution_column="genres",
        p_t_u_all_users=None,
        p_t_i_all_items=None,
         popularity_bins=None,
        movie_budget=None,
        movies_bins=None,
    ):
        RMC = 0
        N = len(recomended_items)

        if user_id not in p_t_u_all_users:
            p_g_u = Metrics.get_p_t_u_distribution(
                trainratings,
                movies,
                user_id,
                weight="rating",
                distribution_column=distribution_column,
            )
        else:
            p_g_u = p_t_u_all_users[user_id]

        if fairness_function.lower() == "kl":
            void = Metrics.get_user_KL_divergence(
                trainratings,
                movies,
                user_id,
                [],
                p_g_u=p_g_u,
                distribution_column=distribution_column,
                p_t_i_all_items=p_t_i_all_items,
                p_t_u_all_users=p_t_u_all_users,
                popularity_bins=popularity_bins,
                movie_budget=movie_budget,
                movies_bins=movies_bins,
            )

        for i in range(1, N+1):
            partial_recomendation = recomended_items[:i]
            RMC += Metrics.get_miscalibration(
                trainratings,
                movies,
                void,
                user_id,
                partial_recomendation,
                fairness_function,
                p_g_u=p_g_u,
                distribution_column=distribution_column,
                p_t_u_all_users=p_t_u_all_users,
                p_t_i_all_items=p_t_i_all_items,
                popularity_bins=popularity_bins,
                movie_budget=movie_budget,
                movies_bins=movies_bins,
            )
        if N == 0: return 0 
        return RMC / N

    @staticmethod
    def get_user_calibration_error(
        movies,
        user_id,
        recomended_items,
        p_g_u,
        distribution_column="genres",
        p_t_i_all_items=None,
    ):
        q_g_u = Metrics.get_q_t_u_distribution(
            recomended_items,
            movies,
            distribution_column=distribution_column,
            p_t_i_all_items=p_t_i_all_items,
        )

        error = [
            abs(p_g_u.get(genre, 0) - q_g_u.get(genre, 0))
            for genre in q_g_u.keys()
        ]
        if len(p_g_u.keys()) != 0:
            return sum(error) / len(p_g_u.keys())
        return 0

    @staticmethod
    def get_user_average_calibration_error(
        movies,
        trainratings,
        user_id,
        recomended_items,
        distribution_column="genres",
        p_t_u_all_users=None,
        p_t_i_all_items=None,
    ):
        ACE = 0
        N = len(recomended_items)

        if p_t_u_all_users is None:
            p_g_u = Metrics.get_p_t_u_distribution(
                trainratings,
                movies,
                user_id,
                weight="rating",
                distribution_column=distribution_column,
            )
        else:
            if user_id not in p_t_u_all_users:
                p_g_u = Metrics.get_p_t_u_distribution(
                trainratings,
                movies,
                user_id,
                weight="rating",
                distribution_column=distribution_column,
            )
            else:
                p_g_u = p_t_u_all_users[user_id]

        for i in range(1, N+1):
            partial_recomendation = recomended_items[:i]
            ACE += Metrics.get_user_calibration_error(
                movies,
                user_id,
                partial_recomendation,
                p_g_u=p_g_u,
                distribution_column=distribution_column,
                p_t_i_all_items=p_t_i_all_items,
            )
        if N == 0: return 0
        return ACE / N

    @staticmethod
    def get_mean_average_calibration_error(
        movies,
        trainratings,
        recomendation_list_per_user,
        distribution_column="genres",
        p_t_u_all_users=None,
        p_t_i_all_items=None,
    ):

        MACE = 0

        for user_id in recomendation_list_per_user.keys():
            MACE += Metrics.get_user_average_calibration_error(
                movies,
                trainratings,
                user_id,
                recomendation_list_per_user[user_id],
                distribution_column=distribution_column,
                p_t_u_all_users=p_t_u_all_users,
                p_t_i_all_items=p_t_i_all_items,
            )

        if len(recomendation_list_per_user.keys()) == 0: return 0
        return MACE / len(recomendation_list_per_user.keys())

    @staticmethod
    def long_tail_coverage(recomendations_per_user, all_items, isPairwise=False):
        m_t_items = all_items[all_items["popularity"].isin(["M", "T"])]

        subset_m_t = list(m_t_items["item"].unique())

        union_recommended_m_t = []

        for user in recomendations_per_user:
            for item, score in recomendations_per_user[user]:
                if (isPairwise):
                    if np.any(subset_m_t == item):
                        union_recommended_m_t.append(item)
                else:
                    if item in subset_m_t:
                        union_recommended_m_t.append(item)

        return len(set(union_recommended_m_t)) / len(m_t_items)

    @staticmethod
    def aggregate_diversity(recomendations_per_user, all_items, isPairwise=False):
        set_recommended_items = []

        for user in recomendations_per_user:
            set_recommended_items = set_recommended_items + [
                i[0] for i in recomendations_per_user[user]
            ]

        if(isPairwise):
            set_recommended_items = set(tuple(item) for item in set_recommended_items)
        else:
            set_recommended_items = set(set_recommended_items)
        

        return len(set_recommended_items) / len(all_items)

    @staticmethod
    def prec(recomendations_per_user, testratings, isPairwise=False, n=10):
        n_relev = 0
        n_ = 0
        for user in recomendations_per_user:
            n_relevant = 0
            aux = testratings[testratings["user"] == user]
            y_pred = sorted(recomendations_per_user[user], key=lambda x: x[1])[:n+1]

            for item, index in y_pred:
                n_ += 1
                if(isPairwise):
                    if np.any(np.isin(item, aux["item"])):
                        n_relevant += 1
                else:
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
        distribution_column="genres",
        p_g_u_all_users=None,
    ):
        if user not in p_g_u_all_users:
            p_g_u = Metrics.get_p_t_u_distribution(
                trainratings_data,
                movies_data,
                user,
                distribution_column=distribution_column,
            )
        else:
            p_g_u = p_g_u_all_users[user]
        return len(p_g_u) / len(all_genres)

    def tradeoff_variance(
        self,
        all_genres,
        trainratings_data,
        movies_data,
        user,
        distribution_column="genres",
        p_g_u_all_users=None,
    ):
        if p_g_u_all_users is not None:
            if user not in p_g_u_all_users:
                p_g_u = Metrics.get_p_t_u_distribution(
                    trainratings_data,
                    movies_data,
                    user,
                    distribution_column=distribution_column,
                )
            else:
                p_g_u = p_g_u_all_users[user]

            if isinstance(all_genres, list):
                mean_genre = 0
                for genre in p_g_u:
                    mean_genre += p_g_u[genre]
                mean_genre /= len(all_genres)

                variance = 0

                for genre in all_genres:
                    variance += (p_g_u.get(genre, 0) - mean_genre) ** 2

                return 1 - (variance / len(all_genres))

    @staticmethod
    def gini_index(recomendations_per_user, all_items, all_users):
        ...

    @staticmethod
    def item_is_relevant(testratings, user_id, item_id, isPairwise = False):
        aux = testratings[testratings["user"] == user_id]

        if (isPairwise):
            if np.any(item_id == np.array(aux['item'])):
                return True
            return False
        else:
            if item_id in list(aux['item']):
                return True
            return False

    @staticmethod
    def user_relevant_items(testratings, user_id):
        aux = testratings[testratings["user"] == user_id]
        return len(aux['item'].unique())

    @staticmethod
    def ap_at(testratings, user_id, recommendation_list, isPairwise=False, N=10):
        AP = 0

        for k in range(N):
            # if Metrics.item_is_relevant(testratings, user_id, recommendation_list[k][0]):
            AP += Metrics.prec({user_id: recommendation_list}, testratings=testratings, isPairwise=isPairwise, n=k)
        if N == 0: return 0
        return AP / N


    @staticmethod
    def map_at(testratings, recomendations_per_user, isPairwise=False, N=10):
        MAP = 0
        for user_id in recomendations_per_user:
            AP = Metrics.ap_at(testratings=testratings, user_id=user_id, recommendation_list=recomendations_per_user[user_id], isPairwise=isPairwise, N=N)
                        
            MAP += AP
        if len(recomendations_per_user.keys()) != 0:
            return MAP / len(recomendations_per_user.keys())
        return 0

    @staticmethod
    def mrr(testratings, recomendations_per_user, isPairWise = False):
        MRR = 0

        for user_id in recomendations_per_user:
            user_find = False
            user_mrr = 0
            for item_id, index in recomendations_per_user[user_id]:
                if user_find == False:
                    if Metrics.item_is_relevant(testratings, user_id=user_id, item_id=item_id, isPairwise=isPairWise):
                        MRR += (1/index)
                        user_mrr += (1/index)
                        user_find = True
                        break
                else:
                    break
        if len(recomendations_per_user.keys()) != 0:    
            return MRR / len(recomendations_per_user.keys())
        return 0


        

