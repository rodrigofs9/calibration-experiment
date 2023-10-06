"""
Bayesian Personalized Ranking
Matrix Factorization model and a variety of classes
implementing different sampling strategies.
"""
import pandas as pd
import numpy as np
from math import exp
import random
import numpy as np
from tqdm import trange
import pandas as pd
from scipy.sparse import csr_matrix

class BPRArgs(object):

    def __init__(self,learning_rate=0.05,
                 bias_regularization=1.0,
                 user_regularization=0.0025,
                 positive_item_regularization=0.0025,
                 negative_item_regularization=0.00025,
                 update_negative_item_factors=True):
        self.learning_rate = learning_rate
        self.bias_regularization = bias_regularization
        self.user_regularization = user_regularization
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization
        self.update_negative_item_factors = update_negative_item_factors

class BPR(object):

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
                             .groupby('user_id')
                             .agg(**{'num_items': ('item_id', 'count')})
                             .reset_index())
        data = data.merge(data_user_num_items, on='user_id', how='inner')
        data = data[data['num_items'] > 1]

        for col in (items_col, users_col, ratings_col):
            data[col] = data[col].astype('category')

        ratings = csr_matrix((data[ratings_col],
                              (data[users_col].cat.codes, data[items_col].cat.codes)))
        ratings.eliminate_zeros()
        return ratings, data

    def __init__(self,D,args):
        """initialise BPR matrix factorization model
        D: number of factors
        """
        self.D = D
        self.learning_rate = args.learning_rate
        self.bias_regularization = args.bias_regularization
        self.user_regularization = args.user_regularization
        self.positive_item_regularization = args.positive_item_regularization
        self.negative_item_regularization = args.negative_item_regularization
        self.update_negative_item_factors = args.update_negative_item_factors


    def fit(self, user_item_train_df):

        train = pd.DataFrame(
            list(user_item_train_df.all_ratings()),
            columns=['user_id', 'item_id', 'rating']
        )
        
        from scipy.sparse import csr_matrix

        items_col = 'item_id'
        users_col = 'user_id'
        ratings_col = 'rating'
        threshold = 0
        X, df = self.create_matrix(train, users_col, items_col, ratings_col, threshold)

        data = X

        sample_negative_items_empirically = True
        sampler = UniformPairWithoutReplacement(sample_negative_items_empirically)
        num_iters = 10
        self.train(data,sampler,num_iters)

    def train(self,data,sampler,num_iters):
        """train model
        data: user-item matrix as a scipy sparse matrix
              users and items are zero-indexed
        """
        self.init(data)

        print ('initial loss = {0}'.format(self.loss()))
        for it in xrange(num_iters):
            print ('starting iteration {0}'.format(it))
            for u,i,j in sampler.generate_samples(self.data):
                self.update_factors(u,i,j)
            print ('iteration {0}: loss = {1}'.format(it,self.loss()))

    def init(self,data):
        self.data = data
        self.num_users,self.num_items = self.data.shape

        self.item_bias = np.zeros(self.num_items)
        self.user_factors = np.random.random_sample((self.num_users,self.D))
        self.item_factors = np.random.random_sample((self.num_items,self.D))

        self.create_loss_samples()

    def create_loss_samples(self):
        # apply rule of thumb to decide num samples over which to compute loss
        num_loss_samples = int(100*self.num_users**0.5)

        print ('sampling {0} <user,item i,item j> triples...'.format(num_loss_samples))
        sampler = UniformUserUniformItem(True)
        self.loss_samples = [t for t in sampler.generate_samples(data,num_loss_samples)]

    def update_factors(self,u,i,j,update_u=True,update_i=True):
        """apply SGD update"""
        update_j = self.update_negative_item_factors

        x = self.item_bias[i] - self.item_bias[j] \
            + np.dot(self.user_factors[u,:],self.item_factors[i,:]-self.item_factors[j,:])

        z = 1.0/(1.0+exp(x))

        # update bias terms
        if update_i:
            d = z - self.bias_regularization * self.item_bias[i]
            self.item_bias[i] += self.learning_rate * d
        if update_j:
            d = -z - self.bias_regularization * self.item_bias[j]
            self.item_bias[j] += self.learning_rate * d

        if update_u:
            d = (self.item_factors[i,:]-self.item_factors[j,:])*z - self.user_regularization*self.user_factors[u,:]
            self.user_factors[u,:] += self.learning_rate*d
        if update_i:
            d = self.user_factors[u,:]*z - self.positive_item_regularization*self.item_factors[i,:]
            self.item_factors[i,:] += self.learning_rate*d
        if update_j:
            d = -self.user_factors[u,:]*z - self.negative_item_regularization*self.item_factors[j,:]
            self.item_factors[j,:] += self.learning_rate*d

    def loss(self):
        ranking_loss = 0;
        for u,i,j in self.loss_samples:
            x = self.predict(u,i) - self.predict(u,j)
            ranking_loss += 1.0/(1.0+exp(x))

        complexity = 0;
        for u,i,j in self.loss_samples:
            complexity += self.user_regularization * np.dot(self.user_factors[u],self.user_factors[u])
            complexity += self.positive_item_regularization * np.dot(self.item_factors[i],self.item_factors[i])
            complexity += self.negative_item_regularization * np.dot(self.item_factors[j],self.item_factors[j])
            complexity += self.bias_regularization * self.item_bias[i]**2
            complexity += self.bias_regularization * self.item_bias[j]**2

        return ranking_loss + 0.5*complexity

    def predict(self,u,i):
        return self.item_bias[i] + np.dot(self.user_factors[u],self.item_factors[i])

    def _predict_user(self, user):
        """
        returns the predicted ratings for the specified user,
        this is mainly used in computing evaluation metric
        """
        user_pred = self.user_factors[user].dot(self.item_factors.T)
        return user_pred

    def test(self, data):
        user_id = int(data[0][0])

        pred = [i for i in self._predict_user(user_id)]

        return list(enumerate(pred))


# sampling strategies

class Sampler(object):

    def __init__(self,sample_negative_items_empirically):
        self.sample_negative_items_empirically = sample_negative_items_empirically

    def init(self,data,max_samples=None):
        self.data = data
        self.num_users,self.num_items = data.shape
        self.max_samples = max_samples

    def sample_user(self):
        u = self.uniform_user()
        num_items = self.data[u].getnnz()
        assert(num_items > 0 and num_items != self.num_items)
        return u

    def sample_negative_item(self,user_items):
        j = self.random_item()
        while j in user_items:
            j = self.random_item()
        return j

    def uniform_user(self):
        return random.randint(0,self.num_users-1)

    def random_item(self):
        """sample an item uniformly or from the empirical distribution
           observed in the training data
        """
        if self.sample_negative_items_empirically:
            # just pick something someone rated!
            u = self.uniform_user()
            i = random.choice(self.data[u].indices)
        else:
            i = random.randint(0,self.num_items-1)
        return i

    def num_samples(self,n):
        if self.max_samples is None:
            return n
        return min(n,self.max_samples)

class UniformUserUniformItem(Sampler):

    def generate_samples(self,data,max_samples=None):
        self.init(data,max_samples)
        for _ in xrange(self.num_samples(self.data.nnz)):
            u = self.uniform_user()
            # sample positive item
            i = random.choice(self.data[u].indices)
            j = self.sample_negative_item(self.data[u].indices)
            yield u,i,j

class UniformUserUniformItemWithoutReplacement(Sampler):

    def generate_samples(self,data,max_samples=None):
        self.init(self,data,max_samples)
        # make a local copy of data as we're going to "forget" some entries
        self.local_data = self.data.copy()
        for _ in xrange(self.num_samples(self.data.nnz)):
            u = self.uniform_user()
            # sample positive item without replacement if we can
            user_items = self.local_data[u].nonzero()[1]
            if len(user_items) == 0:
                # reset user data if it's all been sampled
                for ix in self.local_data[u].indices:
                    self.local_data[u,ix] = self.data[u,ix]
                user_items = self.local_data[u].nonzero()[1]
            i = random.choice(user_items)
            # forget this item so we don't sample it again for the same user
            self.local_data[u,i] = 0
            j = self.sample_negative_item(user_items)
            yield u,i,j

class UniformPair(Sampler):

    def generate_samples(self,data,max_samples=None):
        self.init(data,max_samples)
        for _ in xrange(self.num_samples(self.data.nnz)):
            idx = random.randint(0,self.data.nnz-1)
            u = self.users[self.idx]
            i = self.items[self.idx]
            j = self.sample_negative_item(self.data[u])
            yield u,i,j

class UniformPairWithoutReplacement(Sampler):

    def generate_samples(self,data,max_samples=None):
        self.init(data,max_samples)
        idxs = range(self.data.nnz)
        random.shuffle(idxs)
        self.users,self.items = self.data.nonzero()
        self.users = self.users[idxs]
        self.items = self.items[idxs]
        self.idx = 0
        for _ in xrange(self.num_samples(self.data.nnz)):
            u = self.users[self.idx]
            i = self.items[self.idx]
            j = self.sample_negative_item(self.data[u])
            self.idx += 1
            yield u,i,j

class ExternalSchedule(Sampler):

    def __init__(self,filepath,index_offset=0):
        self.filepath = filepath
        self.index_offset = index_offset

    def generate_samples(self,data,max_samples=None):
        self.init(data,max_samples)
        f = open(self.filepath)
        samples = [map(int,line.strip().split()) for line in f]
        random.shuffle(samples)  # important!
        num_samples = self.num_samples(len(samples))
        for u,i,j in samples[:num_samples]:
            yield u-self.index_offset,i-self.index_offset,j-self.index_offset


import sys
from scipy.io import mmread

    