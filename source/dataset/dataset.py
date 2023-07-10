import pandas as pd
import ast
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

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

            frac = 0.01  # Fração desejada do conjunto de dados (por exemplo, 50%)
            original_df = pd.read_csv(
                f"{path}/test_{index}.csv"
            )
            #original_df = pd.concat([pd.read_csv(
            #    f"{path}/test_{index}.csv"
            #), pd.read_csv(
            #    f"{path}/test_{int(index) - 1}.csv"
            #)])
            original_df.columns = ["user", "item", "rating"]
            #df_sampled = original_df.sample(frac=frac, random_state=int(index))
            self.ratings = original_df
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
        elif type == 'yahoo':
            items_path = f"{path}/items.csv"

            self.ratings = pd.read_csv(
                f"{path}/ratings.csv"
            )
            self.ratings.columns = ["user", "item", "rating"]

            train, test = train_test_split(
                self.ratings, test_size=0.3, random_state=int(index)
            )
            self.test = test
            self.train = train

            self.items = pd.read_csv(items_path, sep=',')
            self.items.columns = ['item', 'title', 'genres']
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


