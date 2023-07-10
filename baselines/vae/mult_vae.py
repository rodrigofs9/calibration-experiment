# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from baselines.vae.ndcg import ndcg_at_k
import pandas as pd
import tensorflow as tf
from baselines.vae.sparse import AffinityMatrix
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Lambda, Dropout
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, Callback
from tempfile import TemporaryDirectory
from baselines.vae.vae_utils import binarize
from baselines.vae.splitters import numpy_stratified_split

# top k items to recommend
TOP_K = 100

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '1m'

# Model parameters
HELDOUT_USERS = 600 # CHANGE FOR DIFFERENT DATASIZE
INTERMEDIATE_DIM = 200
LATENT_DIM = 70
EPOCHS = 400
BATCH_SIZE = 100

# temporary Path to save the optimal model's weights
tmp_dir = TemporaryDirectory()
#WEIGHTS_PATH = os.path.join(tmp_dir, "mvae_weights.hdf5")

SEED = 98765

class LossHistory(Callback):
    """This class is used for saving the validation loss and the training loss per epoch."""

    def on_train_begin(self, logs={}):
        """Initialise the lists where the loss of training and validation will be saved."""
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        """Save the loss of training and validation set at the end of each epoch."""
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))


class Metrics(Callback):
    """Callback function used to calculate the NDCG@k metric of validation set at the end of each epoch.
    Weights of the model with the highest NDCG@k value is saved."""

    def __init__(self, model, val_tr, val_te, mapper, k, save_path=None):

        """Initialize the class parameters.

        Args:
            model: trained model for validation.
            val_tr (numpy.ndarray, float): the click matrix for the validation set training part.
            val_te (numpy.ndarray, float): the click matrix for the validation set testing part.
            mapper (AffinityMatrix): the mapper for converting click matrix to dataframe.
            k (int): number of top k items per user (optional).
            save_path (str): Default path to save weights.
        """
        # Model
        self.model = model

        # Initial value of NDCG
        self.best_ndcg = 0.0

        # Validation data: training and testing parts
        self.val_tr = val_tr
        self.val_te = val_te

        # Mapper for converting from sparse matrix to dataframe
        self.mapper = mapper

        # Top k items to recommend
        self.k = k

        # Options to save the weights of the model for future use
        self.save_path = save_path

    def on_train_begin(self, logs={}):
        """Initialise the list for validation NDCG@k."""
        self._data = []

    def predict(self, x_test):
        """Get predictions for the test set.

        Args:
            x_test (numpy.ndarray): the click matrix for the test set.
        Returns:
            numpy.ndarray: Predicted ratings for the test set.
        """
        return self.model.predict(x_test)

    def recommend_k_items(self, x, k, remove_seen=True):
        """Returns the top-k items ordered by a relevancy score.
        Obtained probabilities are used as recommendation score.

        Args:
            x (numpy.ndarray, int32): input click matrix.
            k (scalar, int32): the number of items to recommend.

        Returns:
            numpy.ndarray: A sparse matrix containing the top_k elements ordered by their score.

        """
        # obtain scores
        score = self.model.predict(x)

        if remove_seen:
            # if true, it removes items from the train set by setting them to zero
            seen_mask = np.not_equal(x, 0)
            score[seen_mask] = 0

        # get the top k items
        top_items = np.argpartition(-score, range(k), axis=1)[:, :k]

        # get a copy of the score matrix
        score_c = score.copy()

        # set to zero the k elements
        score_c[np.arange(score_c.shape[0])[:, None], top_items] = 0

        # set to zeros all elements other then the k
        top_scores = score - score_c

        return top_scores

    def on_epoch_end(self, batch, logs={}):
        """At the end of each epoch calculate NDCG@k of the validation set.

        If the model performance is improved, the model weights are saved.
        Update the list of validation NDCG@k by adding obtained value

        """
        # recommend top k items based on training part of validation set
        top_k = self.recommend_k_items(x=self.val_tr, k=self.k, remove_seen=True)

        # convert recommendations from sparse matrix to dataframe
        top_k_df = self.mapper.map_back_sparse(top_k, kind="prediction")
        test_df = self.mapper.map_back_sparse(self.val_te, kind="ratings")

        # calculate NDCG@k
        NDCG = ndcg_at_k(test_df, top_k_df, col_prediction="prediction", k=self.k)

        # check if there is an improvement in NDCG, if so, update the weights of the saved model
        if NDCG > self.best_ndcg:
            self.best_ndcg = NDCG

            # save the weights of the optimal model
            if self.save_path is not None:
                self.model.save(self.save_path)

        self._data.append(NDCG)

    def get_data(self):
        """Returns a list of the NDCG@k of the validation set metrics calculated
        at the end of each epoch."""
        return self._data


class AnnealingCallback(Callback):
    """This class is used for updating the value of β during the annealing process.
    When β reaches the value of anneal_cap, it stops increasing."""

    def __init__(self, beta, anneal_cap, total_anneal_steps):

        """Constructor

        Args:
            beta (float): current value of beta.
            anneal_cap (float): maximum value that beta can reach.
            total_anneal_steps (int): total number of annealing steps.
        """
        # maximum value that beta can take
        self.anneal_cap = anneal_cap

        # initial value of beta
        self.beta = beta

        # update_count used for calculating the updated value of beta
        self.update_count = 0

        # total annealing steps
        self.total_anneal_steps = total_anneal_steps

    def on_train_begin(self, logs={}):
        """Initialise a list in which the beta value will be saved at the end of each epoch."""
        self._beta = []

    def on_batch_end(self, epoch, logs={}):
        """At the end of each batch the beta should is updated until it reaches the values of anneal cap."""
        self.update_count = self.update_count + 1

        new_beta = min(
            1.0 * self.update_count / self.total_anneal_steps, self.anneal_cap
        )

        K.set_value(self.beta, new_beta)

    def on_epoch_end(self, epoch, logs={}):
        """At the end of each epoch save the value of beta in _beta list."""
        tmp = K.eval(self.beta)
        self._beta.append(tmp)

    def get_data(self):
        """Returns a list of the beta values per epoch."""
        return self._beta


class Mult_VAE:
    """Multinomial Variational Autoencoders (Multi-VAE) for Collaborative Filtering implementation

    :Citation:

        Liang, Dawen, et al. "Variational autoencoders for collaborative filtering."
        Proceedings of the 2018 World Wide Web Conference. 2018.
        https://arxiv.org/pdf/1802.05814.pdf
    """

    def __init__(
        self,
        n_users,
        original_dim,
        intermediate_dim=200,
        latent_dim=70,
        n_epochs=400,
        batch_size=100,
        k=100,
        verbose=1,
        drop_encoder=0.5,
        drop_decoder=0.5,
        beta=1.0,
        annealing=True,
        anneal_cap=1.0,
        seed=None,
        save_path=None,
    ):

        """Constructor

        Args:
            n_users (int): Number of unique users in the train set.
            original_dim (int): Number of unique items in the train set.
            intermediate_dim (int): Dimension of intermediate space.
            latent_dim (int): Dimension of latent space.
            n_epochs (int): Number of epochs for training.
            batch_size (int): Batch size.
            k (int): number of top k items per user.
            verbose (int): Whether to show the training output or not.
            drop_encoder (float): Dropout percentage of the encoder.
            drop_decoder (float): Dropout percentage of the decoder.
            beta (float): a constant parameter β in the ELBO function,
                  when you are not using annealing (annealing=False)
            annealing (bool): option of using annealing method for training the model (True)
                  or not using annealing, keeping a constant beta (False)
            anneal_cap (float): maximum value that beta can take during annealing process.
            seed (int): Seed.
            save_path (str): Default path to save weights.
        """
        # Seed
        self.seed = seed
        np.random.seed(self.seed)

        # Parameters
        self.n_users = n_users
        self.original_dim = original_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.k = k
        self.verbose = verbose

        # Compute samples per epoch
        self.number_of_batches = self.n_users // self.batch_size

        # Annealing parameters
        self.anneal_cap = anneal_cap
        self.annealing = annealing

        if self.annealing:
            self.beta = K.variable(0.0)
        else:
            self.beta = beta

        # Compute total annealing steps
        self.total_anneal_steps = (
            self.number_of_batches
            * (self.n_epochs - int(self.n_epochs * 0.2))
            // self.anneal_cap
        )

        # Dropout parameters
        self.drop_encoder = drop_encoder
        self.drop_decoder = drop_decoder

        # Path to save optimal model
        self.save_path = save_path

        # Create StandardVAE model
        self._create_model()

    def _create_model(self):
        """Build and compile model."""
        # Encoding
        self.x = Input(shape=(self.original_dim,))
        self.x_ = Lambda(lambda x: K.l2_normalize(x, axis=1))(self.x)
        self.dropout_encoder = Dropout(self.drop_encoder)(self.x_)

        self.h = Dense(
            self.intermediate_dim,
            activation="tanh",
            kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(
                seed=self.seed
            ),
            bias_initializer=tf.compat.v1.keras.initializers.truncated_normal(
                stddev=0.001, seed=self.seed
            ),
        )(self.dropout_encoder)
        self.z_mean = Dense(self.latent_dim)(self.h)
        self.z_log_var = Dense(self.latent_dim)(self.h)

        # Sampling
        self.z = Lambda(self._take_sample, output_shape=(self.latent_dim,))(
            [self.z_mean, self.z_log_var]
        )

        # Decoding
        self.h_decoder = Dense(
            self.intermediate_dim,
            activation="tanh",
            kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(
                seed=self.seed
            ),
            bias_initializer=tf.compat.v1.keras.initializers.truncated_normal(
                stddev=0.001, seed=self.seed
            ),
        )
        self.dropout_decoder = Dropout(self.drop_decoder)
        self.x_bar = Dense(self.original_dim)
        self.h_decoded = self.h_decoder(self.z)
        self.h_decoded_ = self.dropout_decoder(self.h_decoded)
        self.x_decoded = self.x_bar(self.h_decoded_)

        # Training
        self.model = Model(self.x, self.x_decoded)
        self.model.compile(
            optimizer='adam',
            loss=self._get_vae_loss,
        )

    def _get_vae_loss(self, x, x_bar):
        """Calculate negative ELBO (NELBO)."""
        log_softmax_var = tf.nn.log_softmax(x_bar)
        self.neg_ll = -tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=log_softmax_var * x, axis=-1)
        )
        a = tf.keras.backend.print_tensor(self.neg_ll)  # noqa: F841
        # calculate positive Kullback–Leibler divergence  divergence term
        kl_loss = K.mean(
            0.5
            * K.sum(
                -1 - self.z_log_var + K.square(self.z_mean) + K.exp(self.z_log_var),
                axis=-1,
            )
        )

        # obtain negative ELBO
        neg_ELBO = self.neg_ll + self.beta * kl_loss

        return neg_ELBO

    def _take_sample(self, args):
        """Sample epsilon ∼ N (0,I) and compute z via reparametrization trick."""

        """Calculate latent vector using the reparametrization trick.
           The idea is that sampling from N (_mean, _var) is s the same as sampling from _mean+ epsilon * _var
           where epsilon ∼ N(0,I)."""
        # _mean and _log_var calculated in encoder
        _mean, _log_var = args

        # epsilon
        epsilon = K.random_normal(
            shape=(K.shape(_mean)[0], self.latent_dim),
            mean=0.0,
            stddev=1.0,
            seed=self.seed,
        )

        return _mean + K.exp(_log_var / 2) * epsilon

    def nn_batch_generator(self, x_train):
        """Used for splitting dataset in batches.

        Args:
            x_train (numpy.ndarray): The click matrix for the train set, with float values.
        """
        # Shuffle the batch
        np.random.seed(self.seed)
        shuffle_index = np.arange(np.shape(x_train)[0])
        np.random.shuffle(shuffle_index)
        x = x_train[shuffle_index, :]
        y = x_train[shuffle_index, :]

        # Iterate until making a full epoch
        counter = 0
        while 1:
            index_batch = shuffle_index[
                self.batch_size * counter : self.batch_size * (counter + 1)
            ]
            # Decompress batch
            x_batch = x[index_batch, :]
            y_batch = y[index_batch, :]
            counter += 1
            yield (np.array(x_batch), np.array(y_batch))

            # Stopping rule
            if counter >= self.number_of_batches:
                counter = 0

    def fit(self, trainset):
        """Fit model with the train set.

        Args:
            trainset (surprise.Trainset): The trainset object from Surprise.
        """
        # Obtain both usercount and itemcount after filtering
        usercount = trainset.n_users
        itemcount = trainset.n_items

        # Compute sparsity after filtering
        sparsity = trainset.global_mean

        print("After filtering, there are %d watching events from %d users and %d items (sparsity: %.3f%%)" %
            (trainset.n_ratings, usercount, itemcount, sparsity * 100))

        unique_users = trainset.all_users()
        np.random.seed(SEED)
        unique_users = np.random.permutation(unique_users)

        # Create train/validation/test users
        n_users = len(unique_users)
        print("Number of unique users:", n_users)

        train_users = unique_users[:(n_users - HELDOUT_USERS * 2)]
        print("\nNumber of training users:", len(train_users))

        val_users = unique_users[(n_users - HELDOUT_USERS * 2):(n_users - HELDOUT_USERS)]
        print("\nNumber of validation users:", len(val_users))

        test_users = unique_users[(n_users - HELDOUT_USERS):]
        print("\nNumber of test users:", len(test_users))

        # For training set keep only users that are in train_users list
        train_set = [trainset.ur[user_id] for user_id in train_users if user_id in trainset.ur]
        print("Number of training observations: ", sum(len(user_ratings) for user_ratings in train_set))

        # For validation set keep only users that are in val_users list
        val_set = [trainset.ur[user_id] for user_id in val_users]
        print("\nNumber of validation observations: ", sum(len(user_ratings) for user_ratings in val_set))

        # For test set keep only users that are in test_users list
        test_set = [trainset.ur[user_id] for user_id in test_users]
        print("\nNumber of test observations: ", sum(len(user_ratings) for user_ratings in test_set))

        # Obtain list of unique items used in training set
        unique_train_items = set()
        for user_ratings in train_set:
            unique_train_items.update(item_id for item_id, _ in user_ratings)
        unique_train_items = sorted(unique_train_items)
        print("Number of unique items that were rated in the training set:", len(unique_train_items))

        # For validation set keep only items that were used in the training set
        val_set = [[(item_id, rating) for item_id, rating in user_ratings if item_id in unique_train_items] for user_ratings in val_set]
        print("Number of validation observations after filtering: ", sum(len(user_ratings) for user_ratings in val_set))

        # For test set keep only items that were used in the training set
        test_set = [[(item_id, rating) for item_id, rating in user_ratings if item_id in unique_train_items] for user_ratings in test_set]
        print("\nNumber of test observations after filtering: ", sum(len(user_ratings) for user_ratings in test_set))

        train_data = []
        for user_id in train_users:
            if user_id in trainset.ur:
                user_ratings = trainset.ur[user_id]
                for item_id, rating in user_ratings:
                    train_data.append({'user_id': user_id, 'item_id': item_id, 'rating': rating})
        train_df = pd.DataFrame(train_data)

        val_data = []
        for user_id in val_users:
            if user_id in trainset.ur:
                user_ratings = trainset.ur[user_id]
                for item_id, rating in user_ratings:
                    val_data.append({'user_id': user_id, 'item_id': item_id, 'rating': rating})
        val_df = pd.DataFrame(val_data)

        test_data = []
        for user_id in test_users:
            if user_id in trainset.ur:
                user_ratings = trainset.ur[user_id]
                for item_id, rating in user_ratings:
                    test_data.append({'user_id': user_id, 'item_id': item_id, 'rating': rating})
        test_df = pd.DataFrame(test_data)

        # Instantiate the sparse matrix generation for train, validation, and test sets
        # Use the list of unique items from the training set for all sets
        am_train = AffinityMatrix(df=train_df, items_list=unique_train_items)

        am_val = AffinityMatrix(df=val_df, items_list=unique_train_items)

        am_test = AffinityMatrix(df=test_df, items_list=unique_train_items)

        # Obtain the sparse matrix for train, validation, and test sets
        train_data, _, _ = am_train.gen_affinity_matrix()

        val_data, val_map_users, val_map_items = am_val.gen_affinity_matrix()
        print(val_data.shape)

        test_data, test_map_users, test_map_items = am_test.gen_affinity_matrix()
        print(test_data.shape)

        val_data_tr, val_data_te = numpy_stratified_split(val_data, ratio=0.75, seed=SEED)
        test_data_tr, test_data_te = numpy_stratified_split(test_data, ratio=0.75, seed=SEED)

        # Binarize train, validation, and test data
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
        for user_id, item_id, rating in trainset.all_ratings():
            if (test_map_users.get(user_id) is not None) and (test_map_items.get(item_id) is not None):
                user_new = test_map_users.get(user_id)
                item_new = test_map_items.get(item_id)
                test_data_te_ratings[user_new, item_new] = rating

            if (val_map_users.get(user_id) is not None) and (val_map_items.get(item_id) is not None):
                user_new = val_map_users.get(user_id)
                item_new = val_map_items.get(item_id)
                val_data_te_ratings[user_new, item_new] = rating

        # test_data_te_ratings
        #val_data_te_ratings = val_data_te_ratings.to_numpy()
        #test_data_te_ratings = test_data_te_ratings.to_numpy()

        # initialise LossHistory used for saving loss of validation and train set per epoch
        history = LossHistory()

        # initialise Metrics  used for calculating NDCG@k per epoch
        # and saving the model weights with the highest NDCG@k value
        x_train = train_data
        x_valid = val_data
        x_val_tr = val_data_tr
        x_val_te = val_data_te_ratings
        mapper = am_val

        metrics = Metrics(
            model=self.model,
            val_tr=x_val_tr,
            val_te=x_val_te,
            mapper=mapper,
            k=self.k,
            save_path=self.save_path,
        )

        self.reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=1, min_lr=0.0001
        )

        if self.annealing:
            # initialise AnnealingCallback for annealing process
            anneal = AnnealingCallback(
                self.beta, self.anneal_cap, self.total_anneal_steps
            )

            # fit model
            self.model.fit(
                x=self.nn_batch_generator(x_train),
                steps_per_epoch=self.number_of_batches,
                epochs=self.n_epochs,
                verbose=str(1),
                callbacks=[metrics, history, self.reduce_lr, anneal],
                validation_data=(x_valid, x_valid),
            )

            self.ls_beta = anneal.get_data()

        else:
            self.model.fit(
                x=self.nn_batch_generator(x_train),
                steps_per_epoch=self.number_of_batches,
                epochs=self.n_epochs,
                verbose=str(1),
                callbacks=[metrics, history, self.reduce_lr],
                validation_data=(x_valid, x_valid),
            )

        # save lists
        self.train_loss = history.losses
        self.val_loss = history.val_losses
        self.val_ndcg = metrics.get_data()

    def get_optimal_beta(self):
        """Returns the value of the optimal beta."""
        if self.annealing:
            # find the epoch/index that had the highest NDCG@k value
            index_max_ndcg = np.argmax(self.val_ndcg)

            # using this index find the value that beta had at this epoch
            return self.ls_beta[index_max_ndcg]
        else:
            return self.beta

    def display_metrics(self):
        """Plots:
        1) Loss per epoch both for validation and train set
        2) NDCG@k per epoch of the validation set
        """
        # Plot setup
        plt.figure(figsize=(14, 5))
        sns.set(style="whitegrid")

        # Plot loss on the left graph
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss, color="b", linestyle="-", label="Train")
        plt.plot(self.val_loss, color="r", linestyle="-", label="Val")
        plt.title("\n")
        plt.xlabel("Epochs", size=14)
        plt.ylabel("Loss", size=14)
        plt.legend(loc="upper left")

        # Plot NDCG on the right graph
        plt.subplot(1, 2, 2)
        plt.plot(self.val_ndcg, color="r", linestyle="-", label="Val")
        plt.title("\n")
        plt.xlabel("Epochs", size=14)
        plt.ylabel("NDCG@k", size=14)
        plt.legend(loc="upper left")

        # Add title
        plt.suptitle("TRAINING AND VALIDATION METRICS HISTORY", size=16)
        plt.tight_layout(pad=2)

    def recommend_k_items(self, x, k, remove_seen=True):
        """Returns the top-k items ordered by a relevancy score.
        Obtained probabilities are used as recommendation score.

        Args:
            x (numpy.ndarray, int32): input click matrix.
            k (scalar, int32): the number of items to recommend.
        Returns:
            numpy.ndarray, float: A sparse matrix containing the top_k elements ordered by their score.
        """
        # return optimal model
        self.model.load_weights(self.save_path)

        # obtain scores
        score = self.model.predict(x)

        if remove_seen:
            # if true, it removes items from the train set by setting them to zero
            seen_mask = np.not_equal(x, 0)
            score[seen_mask] = 0
        # get the top k items
        top_items = np.argpartition(-score, range(k), axis=1)[:, :k]
        # get a copy of the score matrix
        score_c = score.copy()
        # set to zero the k elements
        score_c[np.arange(score_c.shape[0])[:, None], top_items] = 0
        # set to zeros all elements other then the k
        top_scores = score - score_c
        return top_scores

    def ndcg_per_epoch(self):
        """Returns the list of NDCG@k at each epoch."""
        return self.val_ndcg
    
    @staticmethod
    def from_json(json_file_path):
        """Instantiate Mult_VAE class from a JSON file.

        Args:
            json_file_path (str): Path to the JSON file containing the model configuration.
        Returns:
            Mult_VAE: An instance of Mult_VAE initialized with the values from the JSON file.
        """
        import json
        import numpy as np

        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        # Extract training data
        training_data = []
        unique_users = set()
        unique_items = set()
        
        for user_id, items in data.items():
            unique_users.add(user_id)
            for item in items:
                item_id, rating = item
                unique_items.add(item_id)
                training_data.append([user_id, item_id, rating])

        training_data = np.array(training_data)

        # Get the dimensions
        num_users = len(unique_users)
        num_items = len(unique_items)

        original_dim = num_items

        return Mult_VAE(num_users, original_dim)