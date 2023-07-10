from source.dataset.dataset import Dataset
import numpy as np
import pandas as pd


class Popularity:
    @staticmethod
    def calculate_interactions(dataset: Dataset):
        interactions_per_item = dataset.train.groupby("item").size()
        return interactions_per_item

    @staticmethod
    def calculate_interactions_sorted(dataset: Dataset):
        interactions_per_item = dataset.train.groupby("item").size().sort_values(ascending=False)
        return interactions_per_item

    @staticmethod
    def generate_popularity_groups(dataset: Dataset, subdivisions="", division='pareto'):

        interactions_per_item = Popularity.calculate_interactions(dataset)
        interactions_per_item_sorted = Popularity.calculate_interactions_sorted(dataset)

        if division == 'pareto':
            top_20_percent = 0
            top_20_percent_last_interaction = 0
            bottom_20_percent = len(interactions_per_item_sorted)
            bottom_20_percent_last_interaction = 0
            total_interactions = interactions_per_item_sorted.sum()
            sum_ = 0
            while (sum_ <  (total_interactions*0.2)):
                bottom_20_percent-=1
                sum_ = sum(interactions_per_item_sorted.values.tolist()[bottom_20_percent:])
                bottom_20_percent_last_interaction = interactions_per_item_sorted.values.tolist()[bottom_20_percent]
            
            sum_ = 0
            while (sum_ <  (total_interactions*0.2)):
                top_20_percent+=1
                sum_ = sum(interactions_per_item_sorted.values.tolist()[:top_20_percent])
                top_20_percent_last_interaction = interactions_per_item_sorted.values.tolist()[top_20_percent-1]
            
        else:
            ...

        print(f"Divisions {top_20_percent_last_interaction} {bottom_20_percent_last_interaction}")

        if subdivisions == "mean":

            high_pop = interactions_per_item >= top_20_percent_last_interaction
            medium_pop_bool = []
            medium_pop_ids = []

            for w, i, j in zip(
                (interactions_per_item.index),
                (interactions_per_item <= top_20_percent_last_interaction),
                (interactions_per_item >= bottom_20_percent_last_interaction),
            ):
                medium_pop_ids.append(w)
                medium_pop_bool.append((i and j))

            medium_pop = pd.Series(data=medium_pop_bool, index=medium_pop_ids)
            low_pop = interactions_per_item < (np.mean(interactions_per_item))

            popularity_group = []
            pop_index = []
            for i, h, m, l in zip(
                interactions_per_item.index, high_pop, medium_pop, low_pop
            ):
                pop_index.append(i)
                if h:
                    popularity_group.append((i, "H"))
                elif m:
                    popularity_group.append((i, "M"))
                else:
                    popularity_group.append((i, "T"))

            popularity_series = pd.DataFrame(
                data=popularity_group, columns=["item", "popularity"]
            )
            print(popularity_series)
            dataset.items = dataset.items.merge(
                popularity_series, how="inner", on="item"
            )

            print(dataset.items['popularity'].value_counts())

            return dataset
