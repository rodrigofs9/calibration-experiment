import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

import pandas as pd
from baselines.splitters import min_rating_filter_pandas
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, test_size=0.3):
        self.ratings = None
        self.items = None
        self.users = None

        self.ratings_train = None
        self.ratings_test = None
        self.test_size = test_size

    def load_dataset(self, path: str, type = "1ml", index = '0'):
        self.type = type

        sample_frac = 1.0 

        if type == "yahoo_song":
            ratings = pd.read_csv(f"{path}/train_0.csv")
            ratings.columns = ["user", "item", "rating"]
            ratings = ratings.dropna(subset = ['user'])
            ratings = ratings.dropna(subset = ['item'])
            ratings = ratings.dropna(subset = ['rating'])

            df_preferred = ratings[ratings['rating'] > 0.5]

            # Keep users who clicked on at least 10 movies
            df = min_rating_filter_pandas(df_preferred, min_rating = 10, filter_by = "user")

            # Keep movies that were clicked on by at least on 10 user
            df = min_rating_filter_pandas(df, min_rating = 10, filter_by = "item")
            
            unique_users = df['user'].unique()
            user_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_users)}

            unique_items = df['item'].unique()
            item_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_items)}

            df['old_user'] = df['user']
            df['user'] = df['old_user'].map(user_mapping)

            df['old_item'] = df['item']
            df['item'] = df['old_item'].map(item_mapping)

            usercount = df[['user']].groupby('user', as_index = False).size()
            itemcount = df[['item']].groupby('item', as_index = False).size()

            print("After filtering, there are %d watching events from %d users and %d movies" % 
                (df.shape[0], usercount.shape[0], itemcount.shape[0]))
            df = df.sample(frac = sample_frac, random_state = 42)

            self.ratings = df
            train, test = train_test_split(df, test_size = 0.3, random_state = int(index))
            self.test = test
            self.train = train

            items = pd.read_csv(f"{path}/items.csv", sep = ',')
            items.columns = ['item', 'title', 'genres', 'other'] 
            items = items.dropna(subset = ['item'])
            items = items.dropna(subset = ['title'])
            items = items.dropna(subset = ['genres'])
            print("Before filtering, there are %d watching events" % 
                (items.shape[0]))

            df_cleaned = items.dropna(subset=['genres'])
            df_cleaned = df_cleaned.loc[df_cleaned['genres'] != '(no genres listed)']
            df_cleaned = df_cleaned.loc[df_cleaned['genres'] != 'Unknown']
            df_filtered = df_cleaned[df_cleaned['item'].isin(df['item'])]
            print("After filtering, there are %d watching events" % 
                (df_filtered.shape[0]))

            self.items = df_filtered
        elif type == "yahoo_movies":
            ratings = pd.read_csv(f"{path}/ratings.csv")
            ratings.columns = ["user", "item", "rating"]
            ratings = ratings.dropna(subset = ['user'])
            ratings = ratings.dropna(subset = ['item'])
            ratings = ratings.dropna(subset = ['rating'])

            df = ratings

            unique_items = df['item'].unique()
            item_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_items)}

            usercount = df[['user']].groupby('user', as_index = False).size()
            itemcount = df[['item']].groupby('item', as_index = False).size()
            print("Before filtering, there are %d watching events from %d users and %d movies" % 
                (df.shape[0], usercount.shape[0], itemcount.shape[0]))

            df['old_item'] = df['item']
            df['item'] = df['old_item'].map(item_mapping)

            # Obtain both usercount and itemcount after filtering
            usercount = df[['user']].groupby('user', as_index = False).size()
            itemcount = df[['item']].groupby('item', as_index = False).size()

            print("After filtering, there are %d watching events from %d users and %d movies" % 
                (df.shape[0], usercount.shape[0], itemcount.shape[0]))
            
            df = df.sample(frac = sample_frac, random_state = 42)
            self.ratings = df
            print(self.ratings)
            train, test = train_test_split(df, test_size = 0.3, random_state = int(index))
            self.test = test
            self.train = train

            items = pd.read_csv(f"{path}/items.csv", sep=',')
            items.columns = ['old_item', 'title', 'genres']

            print("Before filtering, there are %d watching events" % 
                (items.shape[0]))
            items['item'] = items['old_item'].map(item_mapping)
            items = items.dropna(subset=['item'])
            items = items.dropna(subset=['title'])
            items = items.dropna(subset=['genres'])

            print(items)

            print("After filtering, there are %d watching events" % 
                (items.shape[0]))

            self.items = items
        elif type == 'yahoo_movies_old':
            items_path = f"{path}/items.csv"

            df = pd.read_csv(
                f"{path}/ratings.csv"
            )
            df.columns = ["user", "item", "rating"]

            unique_users = df['user'].unique()
            unique_items = df['item'].unique()

            usercount = df[['user']].groupby('user', as_index = False).size()
            itemcount = df[['item']].groupby('item', as_index = False).size()

            print("After filtering, there are %d watching events from %d users and %d movies" % 
                (df.shape[0], usercount.shape[0], itemcount.shape[0]))

            self.ratings = df
            self.ratings.columns = ["user", "item", "rating"]
            print(self.ratings)
            train, test = train_test_split(df, test_size = 0.3, random_state = int(index))
            
            self.test = test
            self.train = train

            self.items = pd.read_csv(items_path, sep = ',')
            self.items.columns = ['item', 'title', 'genres']
            print(self.items)   
        elif type == "movielens":
            ratings = pd.read_csv(f"{path}/ratings.csv")
            ratings.columns = ["user", "item", "rating", "timestamp"]
            df_preferred = ratings[ratings['rating'] > 3.5]

            # Keep users who clicked on at least 200 movies
            df = min_rating_filter_pandas(df_preferred, min_rating = 200, filter_by = "user")

            # Keep movies that were clicked on by at least on 10 user
            df = min_rating_filter_pandas(df, min_rating = 10, filter_by = "item")

            df = df.sample(frac = 0.02, random_state = int(index))

            unique_users = df['user'].unique()
            user_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_users)}

            unique_items = df['item'].unique()
            item_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_items)}

            df['old_user'] = df['user']
            df['user'] = df['old_user'].map(user_mapping)

            df['old_item'] = df['item']
            df['item'] = df['old_item'].map(item_mapping)

            usercount = df[['user']].groupby('user', as_index = False).size()
            itemcount = df[['item']].groupby('item', as_index = False).size()

            print("After filtering, there are %d watching events from %d users and %d movies" % 
                (df.shape[0], usercount.shape[0], itemcount.shape[0]))
            
            df = df.sample(frac = sample_frac, random_state = 42)
            self.ratings = df
            train, test = train_test_split(df, test_size = 0.3, random_state = int(index))
            self.test = test
            self.train = train

            items = pd.read_csv(f"{path}/items.csv", sep = ',')
            items.columns = ['item', 'title', 'genres'] 
            print("Before filtering, there are %d watching events" % 
                (items.shape[0]))

            df_cleaned = items.dropna(subset = ['genres'])
            df_cleaned = df_cleaned.loc[df_cleaned['genres'] != '(no genres listed)']
            df_filtered = df_cleaned[df_cleaned['item'].isin(df['item'])]
            print("After filtering, there are %d watching events" % 
                (df_filtered.shape[0]))

            self.items = df_filtered
