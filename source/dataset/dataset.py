import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

import pandas as pd
import ast
from sklearn import preprocessing
from baselines.vae.splitters import min_rating_filter_pandas
from sklearn.model_selection import train_test_split
import numpy as np

class Dataset:
    def __init__(self, test_size=0.3):
        self.ratings = None
        self.items = None
        self.users = None

        self.ratings_train = None
        self.ratings_test = None
        self.test_size = test_size

    def clean_dataset(self, rating = 4, users = 500, movies=20, binarize=False):
        # Cut off Ratings
        self.train = self.train.drop(
            self.train[self.train["rating"] < rating].index
        )
        if binarize:
            self.train['rating'] = 1

        self.test = self.test.drop(
            self.test[self.test["rating"] < rating].index
        )
        if binarize:
            self.test['rating'] = 1
        
        # Remove users
        to_remove_users = []
        for index, row in (
            self.train.groupby(["user"], as_index=False).count().iterrows()
        ):
            if row < users:
                to_remove_users.append(index)

        self.train = self.train[
            ~self.train["user"].isin(to_remove_users)
        ]
        self.test = self.test[
            ~self.test["user"].isin(to_remove_users)
        ]

        # Remove items
        to_remove_items = []
        for index, row in (
            self.train.groupby(["item"], as_index=False).count().iterrows()
        ):
            if row < movies:
                to_remove_items.append(index)

        self.train = self.train[
            ~self.train["item"].isin(to_remove_items)
        ]
        self.test = self.test[
            ~self.test["item"].isin(to_remove_items)
        ]

        items_with_interaction = self.train["item"].unique()

        if self.items is not None:
            self.items = self.items[
                self.items["item"].isin(items_with_interaction)
            ]

        self.test = self.test[
            self.test["user"].isin(self.train["user"])
        ]

    def load_local_others_dataset(self, path: str, type=None):
        if type == "BX":
            le = preprocessing.LabelEncoder()
            self.all_data = pd.read_csv(f"{path}/Preprocessed_data.csv")
            self.all_data['book_title'] = le.fit_transform(self.all_data['book_title'])
            self.ratings = self.all_data[['user_id','book_title','rating']]
            self.ratings.columns = ['user', 'item', 'rating']
            
            self.items = self.all_data[["book_title", 'Category']]
            self.items['Category'] = self.items['Category'].apply(lambda x: "|".join(ast.literal_eval(x)) if isinstance(ast.literal_eval(x), list) else x )
            self.items.columns = ['item', 'genres']
    
    def clean_by_user_popularity(self, to_remove_users):
        if self.ratings is not None:
            self.ratings = self.ratings[
                ~self.ratings["user"].isin(to_remove_users)
            ]

    def load_local_movielens_dataset(self, path: str, type="1ml", index='0', movies=5, users=150, binarize=True, rating_=4):
        self.type = type

        if type == 'yahoo_song':
            items_path = f"{path}/items.csv"

            frac = 0.1  # Fração desejada do conjunto de dados (por exemplo, 50%)
            original_df = pd.read_csv(
                f"{path}/test_{index}.csv"
            )
            #original_df = pd.concat([pd.read_csv(
            #    f"{path}/test_{index}.csv"
            #), pd.read_csv(
            #    f"{path}/test_{int(index) - 1}.csv"
            #)])
            original_df.columns = ["user", "item", "rating"]
            df_sampled = original_df.sample(frac=frac, random_state=int(index))
            self.ratings = df_sampled
            self.ratings.columns = ["user", "item", "rating"]

            train, test = train_test_split(
                self.ratings, test_size=0.3, random_state=int(index)
            )
            self.test = test
            self.train = train

            df = pd.read_csv(items_path, sep=',')
            df = df.drop(columns=df.columns[-1]) 
            self.items = df
            self.items.columns = ['item', 'title', 'genres']
            #items_path = f"{path}/items.csv"

            #original_df = pd.read_csv(
            #    f"{path}/test_{index}.csv"
            #)
            #original_df.columns = ["user", "item", "rating"]

            #frac = 0.01  # Fração desejada do conjunto de dados (por exemplo, 50%)
            #df_sampled = original_df.sample(frac=frac, random_state=42)

            # Salve o conjunto de dados amostrado em um novo arquivo, se necessário
            #df_sampled.to_csv(f"{path}/amostra_test_{index}.csv", index=False)

            #self.ratings = df_sampled

            #self.ratings = pd.concat([pd.read_csv(
            #    f"{path}/test_{index}.csv"
            #), pd.read_csv(
            #    f"{path}/train_{index}.csv"
            #)])
            #self.ratings.columns = ["user", "item", "rating"]

            #train, test = train_test_split(
            #    self.ratings, test_size=0.2, random_state=int(index)
            #)
            #self.test = test
            #self.train = train

            #df = pd.read_csv(items_path, sep=',')
            #df = df.drop(columns=df.columns[-1])  
                  
            #self.items = df
            #self.items.columns = ['item', 'title', 'genres']
        elif type == "yahoo-song-new-split":
            ratings = pd.read_csv(
                f"{path}/ratings.csv"
            )
            ratings.columns = ["user", "item", "rating"]
            df_preferred = ratings[ratings['rating'] > 3.5]

            # Keep users who clicked on at least 5 movies
            df = min_rating_filter_pandas(df_preferred, min_rating=10, filter_by="user")

            # Keep movies that were clicked on by at least on 1 user
            df = min_rating_filter_pandas(df, min_rating=10, filter_by="item")

            #df = df.sample(frac=0.2, random_state=int(index))

            unique_users = df['user'].unique()
            user_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_users)}

            unique_items = df['item'].unique()
            item_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_items)}

            # Aplique o mapeamento para criar a coluna 'user' sequencial
            df['old_user'] = df['user']
            df['user'] = df['old_user'].map(user_mapping)

            df['old_item'] = df['item']
            df['item'] = df['old_item'].map(item_mapping)

            # Obtain both usercount and itemcount after filtering
            usercount = df[['user']].groupby('user', as_index = False).size()
            itemcount = df[['item']].groupby('item', as_index = False).size()

            print("After filtering, there are %d watching events from %d users and %d movies" % 
                (df.shape[0], usercount.shape[0], itemcount.shape[0]))
            
            self.ratings = df
            train, test = train_test_split(
                df, test_size=0.3, random_state=int(index)
            )
            self.test = test
            self.train = train

            items = pd.read_csv(f"{path}/items.csv", sep=',')
            items.columns = ['item', 'title', 'genres','other'] 
            print("Before filtering, there are %d watching events" % 
                (items.shape[0]))

            df_cleaned = items.dropna(subset=['genres'])
            df_cleaned = df_cleaned.loc[df_cleaned['genres'] != '(no genres listed)']
            df_filtered = df_cleaned[df_cleaned['item'].isin(df['item'])]
            print("After filtering, there are %d watching events" % 
                (df_filtered.shape[0]))

            self.items = df_filtered
        elif type == "yahoo_song_new":
            ratings = pd.read_csv(f"{path}/train_0.csv")
            ratings.columns = ["user", "item", "rating"]
            ratings = ratings.dropna(subset=['user'])
            ratings = ratings.dropna(subset=['item'])
            ratings = ratings.dropna(subset=['rating'])

            df_preferred = ratings[ratings['rating'] > 0.5]

            # Keep users who clicked on at least 5 movies
            df = min_rating_filter_pandas(df_preferred, min_rating=10, filter_by="user")

            # Keep movies that were clicked on by at least on 1 user
            df = min_rating_filter_pandas(df, min_rating=10, filter_by="item")
            
            unique_users = df['user'].unique()
            user_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_users)}

            unique_items = df['item'].unique()
            item_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_items)}

            # Aplique o mapeamento para criar a coluna 'user' sequencial
            df['old_user'] = df['user']
            df['user'] = df['old_user'].map(user_mapping)

            df['old_item'] = df['item']
            df['item'] = df['old_item'].map(item_mapping)

            # Obtain both usercount and itemcount after filtering
            usercount = df[['user']].groupby('user', as_index = False).size()
            itemcount = df[['item']].groupby('item', as_index = False).size()

            print("After filtering, there are %d watching events from %d users and %d movies" % 
                (df.shape[0], usercount.shape[0], itemcount.shape[0]))
            
            self.ratings = df
            train, test = train_test_split(
                df, test_size=0.3, random_state=int(index)
            )
            self.test = test
            self.train = train

            items = pd.read_csv(f"{path}/items.csv", sep=',')
            items.columns = ['item', 'title', 'genres', 'other'] 
            items = items.dropna(subset=['item'])
            items = items.dropna(subset=['title'])
            items = items.dropna(subset=['genres'])
            print("Before filtering, there are %d watching events" % 
                (items.shape[0]))

            df_cleaned = items.dropna(subset=['genres'])
            df_cleaned = df_cleaned.loc[df_cleaned['genres'] != '(no genres listed)']
            df_cleaned = df_cleaned.loc[df_cleaned['genres'] != 'Unknown']
            df_filtered = df_cleaned[df_cleaned['item'].isin(df['item'])]
            print("After filtering, there are %d watching events" % 
                (df_filtered.shape[0]))

            self.items = df_filtered
        elif type == "yahoo_new":
            ratings = pd.read_csv(f"{path}/ratings.csv")
            ratings.columns = ["user", "item", "rating"]
            ratings = ratings.dropna(subset=['user'])
            ratings = ratings.dropna(subset=['item'])
            ratings = ratings.dropna(subset=['rating'])

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
            
            self.ratings = df
            print(self.ratings)
            train, test = train_test_split(
                df, test_size=0.3, random_state=int(index)
            )
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
        elif type == 'yahoo':
            items_path = f"{path}/items.csv"

            df = pd.read_csv(
                f"{path}/ratings.csv"
            )
            df.columns = ["user", "item", "rating"]

            unique_users = df['user'].unique()
            #user_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_users)}

            unique_items = df['item'].unique()
            #item_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_items)}

            # Aplique o mapeamento para criar a coluna 'user' sequencial
            #df['old_user'] = df['user']
            #df['user'] = df['old_user'].map(user_mapping)

            #df['old_item'] = df['item']
            #df['item'] = df['old_item'].map(item_mapping)

            usercount = df[['user']].groupby('user', as_index = False).size()
            itemcount = df[['item']].groupby('item', as_index = False).size()

            print("After filtering, there are %d watching events from %d users and %d movies" % 
                (df.shape[0], usercount.shape[0], itemcount.shape[0]))

            self.ratings = df
            self.ratings.columns = ["user", "item", "rating"]
            print(self.ratings)
            train, test = train_test_split(
                df, test_size=0.3, random_state=int(index)
            )
            
            self.test = test
            self.train = train

            self.items = pd.read_csv(items_path, sep=',')
            self.items.columns = ['item', 'title', 'genres']
            print(self.items)
        elif type == 'vae':
            items_path = f"{path}/items.csv"

            ratings = pd.read_csv(
                f"{path}/ratings.csv"
            )
            ratings.columns = ["user", "item", "rating"]
            df_preferred = ratings[ratings['rating'] > 3.5]

            # Keep users who clicked on at least 5 movies
            df = min_rating_filter_pandas(df_preferred, min_rating=5, filter_by="user")

            # Keep movies that were clicked on by at least on 1 user
            df = min_rating_filter_pandas(df, min_rating=1, filter_by="item")

            # Obtain both usercount and itemcount after filtering
            usercount = df[['user']].groupby('user', as_index = False).size()
            itemcount = df[['item']].groupby('item', as_index = False).size()

            # Compute sparsity after filtering
            sparsity = 1. * df.shape[0] / (usercount.shape[0] * itemcount.shape[0])

            print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" % 
                (df.shape[0], usercount.shape[0], itemcount.shape[0], sparsity * 100))

            self.ratings = df

            self.items = pd.read_csv(items_path, sep=',')
            self.items.columns = ['item', 'title', 'genres']            
        elif type == 'yahoo_pairwise':
            ratings_path = f"{path}/ratings.csv"
            ratings = pd.read_csv(ratings_path)
            ratings.columns = ["user", "old_item", "rating"]
            
            items_path = f"{path}/items.csv"
            items = pd.read_csv(items_path, sep=',')
            items.columns = ['item', 'title', 'genres', 'other']
            items['item'] = items['old_item'].astype('category').cat.codes
            items['old_item'] = items['old_item'].astype('category')

            ratings = ratings.merge(items[['old_item', 'item']], on='old_item', how='left')

            # Dividir os dados de avaliações em conjuntos de treinamento e teste
            train_ratings, test_ratings = train_test_split(ratings, test_size=0.3, random_state=int(index))

            self.test = train_ratings
            self.train = test_ratings
            self.items = items

            print(ratings.head())
            print(train_ratings.head())
            print(test_ratings.head())
            print(items.head())
        elif type == 'yahoo2':
            items_path = f"{path}/items.csv"

            self.ratings = pd.read_csv(
                f"{path}/ratings2.csv"
            )
            self.ratings.columns = ["user", "item", "rating"]

            train, test = train_test_split(
                self.ratings, test_size=0.2, random_state=int(index)
            )
            self.test = test
            self.train = train

            self.items = pd.read_csv(items_path, sep=',')
            self.items.columns = ['item', 'title', 'genres']
        elif type == 'yahoo_splitted':
            items_path = f"{path}/items.csv"

            self.ratings = pd.read_csv(
                f"{path}/ratings.csv"
            )
            self.ratings.columns = ["user", "item", "rating"]

            self.test = pd.read_csv(
                f"{path}/test_{index}.csv"
            )
            self.train = pd.read_csv(
                f"{path}/train_{index}.csv"
            )

            self.items = pd.read_csv(items_path, sep=',')
            self.items.columns = ['item', 'title', 'genres']
        elif type == 'ml20m_splitted':
            items_path = f"{path}/items.csv"


            self.test = pd.read_csv(
                f"{path}/test_{index}.csv"
            )
            self.train = pd.read_csv(
                f"{path}/train_{index}.csv"
            )

            aux = pd.concat([self.train, self.test])

            self.ratings = aux
            self.ratings.columns = ["user", "item", "rating", "timestamp"]

            self.items = pd.read_csv(items_path, sep=',')
            self.items.columns = ['item', 'title', 'genres']

            with open(f"../data/ml-20m/itemmap_{index}.json") as f:
                import json
                items_map = json.loads(f.read())

            self.items['item'] = self.items['item'].apply(lambda x: items_map.get(str(x), x))

            self.items = self.items[
                self.items["item"].isin(
                    aux['item'].unique().tolist()
                )
            ]
        
        elif type == "1m":
            ratings = pd.read_csv(
                f"{path}/ratings.dat",
                sep="::",
                names=["user", "item", "rating", "timestamp"],
            )
            self.items = pd.read_csv(
                f"{path}/movies.dat",
                sep="::",
                names=['item', 'title', 'genres']
            )
            ratings.columns = ["user", "item", "rating", "timestamp"]

            ratings = ratings.drop(
                ratings[ratings["rating"] < rating_].index
            )
            if binarize:
                ratings['rating'] = 1

            to_remove_users = []
            for index, row in (
                ratings.groupby(["user"], as_index=False).count().iterrows()
            ):
                if row < users:
                    to_remove_users.append(index)

            ratings = ratings[
                ~ratings["user"].isin(to_remove_users)
        ]

            to_remove_items = []
            for index, row in (
                ratings.groupby(["item"], as_index=False).count().iterrows()
            ):
                if row < movies:
                    to_remove_items.append(index)

            ratings = ratings[
                ~ratings["item"].isin(to_remove_items)
            ]
            
            ratings = ratings.sample(frac=1).reset_index(drop=True)

            self.items.columns = ['item', 'title', 'genres']

            train, test = train_test_split(
                ratings, test_size=0.3, random_state=int(index)
            )
            self.test = test
            self.train = train
        elif type == "20m-new":
            ratings = pd.read_csv(
                f"{path}/ratings.csv"
            )
            ratings.columns = ["user", "item", "rating", "timestamp"]
            df_preferred = ratings[ratings['rating'] > 3.5]

            # Keep users who clicked on at least 5 movies
            df = min_rating_filter_pandas(df_preferred, min_rating=200, filter_by="user")

            # Keep movies that were clicked on by at least on 1 user
            df = min_rating_filter_pandas(df, min_rating=10, filter_by="item")

            df = df.sample(frac=0.02, random_state=int(index))

            unique_users = df['user'].unique()
            user_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_users)}

            unique_items = df['item'].unique()
            item_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_items)}

            # Aplique o mapeamento para criar a coluna 'user' sequencial
            df['old_user'] = df['user']
            df['user'] = df['old_user'].map(user_mapping)

            df['old_item'] = df['item']
            df['item'] = df['old_item'].map(item_mapping)

            # Obtain both usercount and itemcount after filtering
            usercount = df[['user']].groupby('user', as_index = False).size()
            itemcount = df[['item']].groupby('item', as_index = False).size()

            print("After filtering, there are %d watching events from %d users and %d movies" % 
                (df.shape[0], usercount.shape[0], itemcount.shape[0]))
            
            self.ratings = df
            train, test = train_test_split(
                df, test_size=0.3, random_state=int(index)
            )
            self.test = test
            self.train = train

            items = pd.read_csv(f"{path}/items.csv", sep=',')
            items.columns = ['item', 'title', 'genres'] 
            print("Before filtering, there are %d watching events" % 
                (items.shape[0]))

            df_cleaned = items.dropna(subset=['genres'])
            df_cleaned = df_cleaned.loc[df_cleaned['genres'] != '(no genres listed)']
            df_filtered = df_cleaned[df_cleaned['item'].isin(df['item'])]
            print("After filtering, there are %d watching events" % 
                (df_filtered.shape[0]))

            self.items = df_filtered
        elif type == "20m":
            ratings = pd.read_csv(
                f"{path}/ratings.csv"
            )
            ratings.columns = ["user", "item", "rating", "timestamp"]

            ratings = ratings.drop(
                ratings[ratings["rating"] < rating_].index
            )
            if binarize:
                ratings['rating'] = 1

            to_remove_users = []
            for index, row in (
                ratings.groupby(["user"], as_index=False).count().iterrows()
            ):
                if (row < users).all():
                    to_remove_users.append(index)

            ratings = ratings[~ratings["user"].isin(to_remove_users)]

            to_remove_items = []
            for index, row in (
                ratings.groupby(["item"], as_index=False).count().iterrows()
            ):
                if (row < movies).all():
                    to_remove_items.append(index)

            ratings = ratings[
                ~ratings["item"].isin(to_remove_items)
            ]
            
            ratings = ratings.sample(frac=1).reset_index(drop=True)

            self.items = pd.read_csv(
                f"{path}/items.csv"
            )
            self.items.columns = ['item', 'title', 'genres']

            train, test = train_test_split(ratings, test_size=0.3, random_state=int(index))

            self.test = test
            self.train = train
        elif type == "good":
            ratings = pd.read_csv(
                f"{path}/ratings.csv"
            )
            ratings.columns = ["user", "item", "rating"]

            self.items = pd.read_csv(
                f"{path}/items.csv"
            )
            self.items.columns = ['item', 'genres']

            train, test = train_test_split(
                ratings, test_size=0.3, random_state=int(index),
                # stratify=ratings['user']
            )
            self.test = test
            self.train = train
        elif type == "100k":
            self.ratings = pd.read_csv(
                f"{path}/u.data",
                sep="\t",
                names=["user", "item", "rating", "timestamp"],
            )
            self.items = pd.read_csv(
                f"{path}/u.item", sep="|", encoding="latin-1"
            )

            self.items.columns = [
                "item",
                "title",
                "release",
                "release2",
                "imdb",
            ] + "unknown|Action|Adventure|Animation|Children's|Comedy|Crime|Documentary|Drama|Fantasy|Film-Noir|Horror|Musical|Mystery|Romance|Sci-Fi|Thriller|War|Western".split(
                "|"
            )
            self.items = self.items[self.items["unknown"] == 0]
            aux = self.items[
                "Action|Adventure|Animation|Children's|Comedy|Crime|Documentary|Drama|Fantasy|Film-Noir|Horror|Musical|Mystery|Romance|Sci-Fi|Thriller|War|Western".split(
                    "|"
                )
            ]

            columns_name = "Action|Adventure|Animation|Children's|Comedy|Crime|Documentary|Drama|Fantasy|Film-Noir|Horror|Musical|Mystery|Romance|Sci-Fi|Thriller|War|Western".split(
                "|"
            )

            new_genre_data = []
            for index, row in aux.iterrows():
                genre = ""
                for ind, columns in enumerate(row):
                    if columns != 0:
                        genre += f"{columns_name[ind]}|"
                new_genre_data.append(genre[:-1])
            new_genre_data

            self.items["genres"] = new_genre_data

    def load_from_csv(self, path: str, sep=","):
        ...



