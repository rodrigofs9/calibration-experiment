import numpy as np
from tqdm import trange
import pandas as pd
from scipy.sparse import csr_matrix
from metrics import Metrics
from scipy.spatial.distance import jensenshannon
from scipy import spatial
import sys
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from itertools import islice 

from tensorboardX import SummaryWriter
import math

class BPR:
    """
    Bayesian Personalized Ranking (BPR) for implicit feedback data

    Parameters
    ----------
    learning_rate : float, default 0.01
        learning rate for gradient descent

    n_factors : int, default 20
        Number/dimension of user and item latent factors

    n_iters : int, default 15
        Number of iterations to train the algorithm
        
    batch_size : int, default 1000
        batch size for batch gradient descent, the original paper
        uses stochastic gradient descent (i.e., batch size of 1),
        but this can make the training unstable (very sensitive to
        learning rate)

    reg : int, default 0.01
        Regularization term for the user and item latent factors

    seed : int, default 1234
        Seed for the randomly initialized user, item latent factors

    verbose : bool, default True
        Whether to print progress bar while training

    Attributes
    ----------
    user_factors : 2d ndarray, shape [n_users, n_factors]
        User latent factors learnt

    item_factors : 2d ndarray, shape [n_items, n_factors]
        Item latent factors learnt

    References
    ----------
    S. Rendle, C. Freudenthaler, Z. Gantner, L. Schmidt-Thieme 
    Bayesian Personalized Ranking from Implicit Feedback
    - https://arxiv.org/abs/1205.2618
    """
    
    def create_matrix(self, data, users_col, items_col, ratings_col, threshold = None):
        """
        creates the sparse user-item interaction matrix,
        if the data is not in the format where the interaction only
        contains the positive items (indicated by 1), then use the 
        threshold parameter to determine which items are considered positive

        Parameters
        ----------
        data : DataFrame
            implicit rating data

        users_col : str
            user column name

        items_col : str
            item column name

        ratings_col : str
            implicit rating column name

        threshold : int, default None
            threshold to determine whether the user-item pair is 
            a positive feedback

        Returns
        -------
        ratings : scipy sparse csr_matrix, shape [n_users, n_items]
            user/item ratings matrix

        data : DataFrame
            implict rating data that retains only the positive feedback
            (if specified to do so)
        """
        if threshold is not None:
            data = data[data[ratings_col] >= threshold]
            data[ratings_col] = 1

        # this ensures each user has at least 2 records to construct a valid
        # train and test split in downstream process, note we might purge
        # some users completely during this process
        data_user_num_items = (data
                             .groupby('user')
                             .agg(**{'num_items': ('item', 'count')})
                             .reset_index())
        data = data.merge(data_user_num_items, on = 'user', how = 'inner')
        data = data[data['num_items'] > 1]

        for col in (items_col, users_col, ratings_col):
            data[col] = data[col].astype('category')

        ratings = csr_matrix((data[ratings_col],
                              (data[users_col].cat.codes, data[items_col].cat.codes)))
        ratings.eliminate_zeros()
        return ratings, data
    
    def __init__(self, learning_rate = 0.01, n_factors = 15, n_iters = 10, 
                 batch_size = 1000, reg = 0.01, seed = 1234, verbose = True,
                 tradeoff = None, distribution_column = None, target_all_users_distribution = None, target_all_items_distribution = None,
                 dataset = None, movies_data = None, tipo = 'gs1'
         ):
        self.tradeoff = tradeoff
        self.distribution_column = distribution_column
        self.target_all_users_distribution = target_all_users_distribution
        self.target_all_items_distribution = target_all_items_distribution
        self.reg = reg
        self.seed = seed
        self.verbose = verbose
        self.n_iters = n_iters
        self.n_factors = n_factors
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.movies_data = movies_data
        self.tipo = tipo
        self.count = 0
        self.count2 = 0
        
        # to avoid re-computation at predict
        self._prediction = None
        
    def test(self, data):
        user_id = int(data[0][0])
        pred = [i for i in self._predict_user(user_id)]

        return list(enumerate(pred))
        
    def fit(self, user_item_train_df):
        self.writer = SummaryWriter()
        distribution_column = self.distribution_column
        target_all_users_distribution = self.target_all_users_distribution
        target_all_items_distribution = self.target_all_items_distribution
        dataset = self.dataset
        movies_data = self.movies_data

        train = pd.DataFrame(
            list(user_item_train_df.all_ratings()),
            columns = ['user', 'item', 'rating']
        )

        items_col = 'item'
        users_col = 'user'
        ratings_col = 'rating'
        threshold = 0
        X, _ = self.create_matrix(train, users_col, items_col, ratings_col, threshold)
        X
        
        ratings = X
        indptr = ratings.indptr
        indices = ratings.indices
        n_users, n_items = ratings.shape
        
        # ensure batch size makes sense, since the algorithm involves
        # for each step randomly sample a user, thus the batch size
        # should be smaller than the total number of users or else
        # we would be sampling the user with replacement
        batch_size = self.batch_size
        if n_users < batch_size:
            batch_size = n_users
            sys.stderr.write('WARNING: Batch size is greater than number of users,'
                             'switching to a batch size of {}\n'.format(n_users))

        batch_iters = n_users // batch_size
        
        # initialize random weights
        rstate = np.random.RandomState(self.seed)
        self.user_factors = rstate.normal(size = (n_users, self.n_factors))
        self.item_factors = rstate.normal(size = (n_items, self.n_factors))
        
        # progress bar for training iteration if verbose is turned on
        loop = range(self.n_iters)
        if self.verbose:
            loop = trange(self.n_iters, desc = self.__class__.__name__)
        
        for _ in loop:
            for _ in range(batch_iters):
                sampled = self._sample(n_users, n_items, indices, indptr)
                sampled_users, sampled_pos_items, sampled_neg_items = sampled
                self._update(
                    sampled_users, 
                    sampled_pos_items,
                    sampled_neg_items,
                    distribution_column = distribution_column,
                    target_all_users_distribution = target_all_users_distribution, 
                    target_all_items_distribution = target_all_items_distribution,
                    dataset = dataset, 
                    movies_data = movies_data
                )
                sampled = self._sample(n_users, n_items, indices, indptr)
                sampled_users, sampled_pos_items, sampled_neg_items = sampled
                self._update2(sampled_users, sampled_pos_items,
                    sampled_neg_items,
                    distribution_column = distribution_column,
                    target_all_users_distribution = target_all_users_distribution, target_all_items_distribution = target_all_items_distribution,
                    dataset = dataset, movies_data = movies_data
                )
        self.writer.close()
        return self
    
    def _sample(self, n_users, n_items, indices, indptr):
        """sample batches of random triplets u, i, j"""
        sampled_pos_items = np.zeros(self.batch_size, dtype = np.int)
        sampled_neg_items = np.zeros(self.batch_size, dtype = np.int)
        sampled_users = np.random.choice(
            n_users, size = self.batch_size, replace = False)

        for idx, user in enumerate(sampled_users):
            pos_items = indices[indptr[user]:indptr[user + 1]]
            pos_item = np.random.choice(pos_items)
            neg_item = np.random.choice(n_items)
            while neg_item in pos_items:
                neg_item = np.random.choice(n_items)

            sampled_pos_items[idx] = pos_item
            sampled_neg_items[idx] = neg_item

        return sampled_users, sampled_pos_items, sampled_neg_items

    def _update2(self, u, i, j, distribution_column, target_all_users_distribution, target_all_items_distribution, dataset, movies_data):
        m = 0
        for user_id, _, jj in zip(u, i, j):
            preds = list(enumerate(self.user_factors[user_id] @ self.item_factors.T))
            preds = sorted(preds, key=lambda x: x[1], reverse=True)
            preds = preds[:10]

            if user_id in target_all_users_distribution:
                KL_ui = Metrics.get_user_KL_divergence(
                    dataset, movies_data,
                    user_id = user_id, recommended_items = preds,
                    target_user_distribution = target_all_users_distribution[user_id], distribution_column = distribution_column,
                    target_all_items_distribution = target_all_items_distribution
                )

                KL_void = Metrics.get_user_KL_divergence(
                    dataset, movies_data,
                    user_id = user_id, recommended_items = [],
                    target_user_distribution = target_all_users_distribution[user_id], distribution_column = distribution_column,
                    target_all_items_distribution = target_all_items_distribution
                )

                # r_uij = np.sum(self.user_factors[uu] * (itemsi - itemsj)).astype(np.float128)
                # r_uij = (div_ui - div_uj).astype(np.float128)
                if KL_void != 0:
                    r_uij = (1 - (KL_ui/KL_void))#*5
                else:
                    r_uij = 0
                sigmoid = r_uij #np.exp(-r_uij) / (1.0 + np.exp(-r_uij))
                m += sigmoid
                #print("KL", KL_ui, KL_void)
                #print("RR", r_uij)
                #print("sigmoid", sigmoid)
                #if math.isnan(sigmoid):
                    #print("Predictions: ", preds)
                    #input()
                
                # sigmoid_tiled = np.tile(sigmoid, (self.n_factors, 1)).T
                for iii, _ in preds:
                    itemsi = self.item_factors[iii]
                    itemsj = self.item_factors[jj]
                    grad_u = sigmoid * (itemsi - itemsj) + self.reg * self.user_factors[user_id]
                    grad_i = sigmoid * self.user_factors[user_id] + self.reg * self.item_factors[iii]
                    self.user_factors[user_id] += self.learning_rate/5 * grad_u
                    self.item_factors[iii] += self.learning_rate/5 * grad_i
        self.count += 1
        m = m/len(u)
        self.writer.add_scalar('MC/update2', m, self.count)
        return self
                
    def _update(self, u, i, j, distribution_column, target_all_users_distribution, target_all_items_distribution, dataset, movies_data):
        """
        update according to the bootstrapped user u, 
        positive item i and negative item j
        """
        m = 0
        for user_id, _, _ in zip(u, i, j):
            preds = list(enumerate(self.user_factors[user_id] @ self.item_factors.T))
            preds = sorted(preds, key = lambda x: x[1], reverse = True)
            preds = preds[:10]

            if user_id in target_all_users_distribution:
                KL_ui = Metrics.get_user_KL_divergence(
                    dataset, 
                    movies_data,
                    user_id = user_id, 
                    recommended_items = preds,
                    target_user_distribution = target_all_users_distribution[user_id], 
                    distribution_column = distribution_column,
                    target_all_items_distribution = target_all_items_distribution
                )

                KL_void = Metrics.get_user_KL_divergence(
                    dataset, 
                    movies_data,
                    user_id = user_id, 
                    recommended_items = [],
                    target_user_distribution = target_all_users_distribution[user_id], 
                    distribution_column = distribution_column,
                    target_all_items_distribution = target_all_items_distribution
                )
                if KL_void != 0: 
                    m += (1-(KL_ui/KL_void))
        m = m/len(u)
        self.count2 += 1
        self.writer.add_scalar('MC/update1', m, self.count2)

        user_u = self.user_factors[u]
        item_i = self.item_factors[i]
        item_j = self.item_factors[j]
        
        # decompose the estimator, compute the difference between
        # the score of the positive items and negative items; a
        # naive implementation might look like the following:
        # r_ui = np.diag(user_u.dot(item_i.T))
        # r_uj = np.diag(user_u.dot(item_j.T))
        # r_uij = r_ui - r_uj
        
        # however, we can do better, so
        # for batch dot product, instead of doing the dot product
        # then only extract the diagonal element (which is the value
        # of that current batch), we perform a hadamard product, 
        # i.e. matrix element-wise product then do a sum along the column will
        # be more efficient since it's less operations
        # http://people.revoledu.com/kardi/tutorial/LinearAlgebra/HadamardProduct.html
        # r_ui = np.sum(user_u * item_i, axis = 1)
        #
        # then we can achieve another speedup by doing the difference
        # on the positive and negative item up front instead of computing
        # r_ui and r_uj separately, these two idea will speed up the operations
        # from 1:14 down to 0.36
        r_uij = np.sum(user_u * (item_i - item_j), axis = 1)
        sigmoid = np.exp(-r_uij) / (1.0 + np.exp(-r_uij))
        
        # repeat the 1 dimension sigmoid n_factors times so
        # the dimension will match when doing the update
        sigmoid_tiled = np.tile(sigmoid, (self.n_factors, 1)).T

        # update using gradient descent
        grad_u = sigmoid_tiled * (item_i - item_j) + self.reg * user_u
        grad_i = sigmoid_tiled * user_u + self.reg * item_i
        grad_j = sigmoid_tiled * -user_u + self.reg * item_j
        self.user_factors[u] += self.learning_rate * grad_u
        self.item_factors[i] += self.learning_rate * grad_i
        self.item_factors[j] += self.learning_rate * grad_j
        return self

    def predict(self):
        """
        Obtain the predicted ratings for every users and items
        by doing a dot product of the learnt user and item vectors.
        The result will be cached to avoid re-computing it every time
        we call predict, thus there will only be an overhead the first
        time we call it. Note, ideally you probably don't need to compute
        this as it returns a dense matrix and may take up huge amounts of
        memory for large datasets
        """
        self._prediction = self.user_factors.dot(self.item_factors.T)

        return self._prediction

    def _predict_user(self, user):
        """
        returns the predicted ratings for the specified user,
        this is mainly used in computing evaluation metric
        """
        user_pred = self.user_factors[user].dot(self.item_factors.T)
        return user_pred

    def recommend(self, ratings, N = 5):
        """
        Returns the top N ranked items for given user id,
        excluding the ones that the user already liked
        
        Parameters
        ----------
        ratings : scipy sparse csr_matrix, shape [n_users, n_items]
            sparse matrix of user-item interactions 
        
        N : int, default 5
            top-N similar items' N
        
        Returns
        -------
        recommendation : 2d ndarray, shape [number of users, N]
            each row is the top-N ranked item for each query user
        """
        n_users = ratings.shape[0]
        recommendation = np.zeros((n_users, N), dtype = np.uint32)
        for user in range(n_users):
            top_n = self._recommend_user(ratings, user, N)
            recommendation[user] = top_n

        return recommendation

    def _recommend_user(self, ratings, user, N):
        """the top-N ranked items for a given user"""
        scores = self._predict_user(user)

        # compute the top N items, removing the items that the user already liked
        # from the result and ensure that we don't get out of bounds error when 
        # we ask for more recommendations than that are available
        liked = set(ratings[user].indices)
        count = N + len(liked)
        if count < scores.shape[0]:

            # when trying to obtain the top-N indices from the score,
            # using argpartition to retrieve the top-N indices in 
            # unsorted order and then sort them will be faster than doing
            # straight up argort on the entire score
            # http://stackoverflow.com/questions/42184499/cannot-understand-numpy-argpartition-output
            ids = np.argpartition(scores, -count)[-count:]
            best_ids = np.argsort(scores[ids])[::-1]
            best = ids[best_ids]
        else:
            best = np.argsort(scores)[::-1]

        top_n = list(islice((rec for rec in best if rec not in liked), N))
        return top_n
    
    def get_similar_items(self, N = 5, item_ids = None):
        """
        return the top N similar items for itemid, where
        cosine distance is used as the distance metric
        
        Parameters
        ----------
        N : int, default 5
            top-N similar items' N
            
        item_ids : 1d iterator, e.g. list or numpy array, default None
            the item ids that we wish to find the similar items
            of, the default None will compute the similar items
            for all the items
        
        Returns
        -------
        similar_items : 2d ndarray, shape [number of query item_ids, N]
            each row is the top-N most similar item id for each
            query item id
        """
        # cosine distance is proportional to normalized euclidean distance,
        # thus we normalize the item vectors and use euclidean metric so
        # we can use the more efficient kd-tree for nearest neighbor search;
        # also the item will always to nearest to itself, so we add 1 to 
        # get an additional nearest item and remove itself at the end
        normed_factors = normalize(self.item_factors)
        knn = NearestNeighbors(n_neighbors = N + 1, metric = 'euclidean')
        knn.fit(normed_factors)

        # returns a distance, index tuple,
        # we don't actually need the distance
        if item_ids is not None:
            normed_factors = normed_factors[item_ids]

        _, items = knn.kneighbors(normed_factors)
        similar_items = items[:, 1:].astype(np.uint32)
        return similar_items