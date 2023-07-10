from sklearn.linear_model import ElasticNet, Ridge, Lasso
import numpy as np
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from baselines.vae.mult_vae import Mult_VAE
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from multiprocessing import Pool
import multiprocessing as multi
from joblib import Parallel, delayed
import json

class SLIM():
    def __init__(self, alpha, l1_ratio, user_num, item_num, lin_model='lasso', path=None):
        if lin_model == 'lasso':
            self.reg = Lasso(alpha=alpha, positive=True)
        elif lin_model == 'elastic':
            self.reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, positive=True)
            
        self.user_num = user_num
        self.item_num = item_num
        self.path = path
           
    def fit(self, user_item_train_df):
        self.load_sim_mat(self.path, user_item_train_df)

    def load_sim_mat(self, path, user_item_train_df):
        self.row = np.array([r[0] for r in user_item_train_df.all_ratings()], dtype=int)
        self.col = np.array([r[1] for r in user_item_train_df.all_ratings()], dtype=int)
        self.data = np.ones(len(list(user_item_train_df.all_ratings())), dtype=int)
        self.rating_mat = csr_matrix((self.data, (self.row, self.col)), shape = (self.user_num, self.item_num))
        
        self.sim_mat = np.loadtxt(path)

        pred_mat = np.dot(self.rating_mat.toarray(), self.sim_mat)
        self.rec_mat = pred_mat - self.rating_mat
      
    def test(self, data):
        user_id = int(data[0][0])
        row_user = self.rec_mat[user_id, :]

        A = np.squeeze(np.asarray(row_user))
        A = list(enumerate(A))

        return A

class VAE:
    def __init__(self, path):
        self.model = path

    def fit(self, aaa):       
        with open(self.model) as f:
            self.recs = json.loads(f.read())
    
    def test(self, data):
        user_id = int(data[0][0])

        return self.recs.get(str(user_id), [])

from scipy import sparse

class SSS:
    def __init__(self, path):
        self.model = path

    def fit(self, aaa):       
        with open(self.model) as f:
            self.recs = json.loads(f.read())
    
    def test(self, data):
        user_id = int(data[0][0])
        return self.recs.get(str(user_id), [])

class NMFL():
    def fit(self, user_item_train_df):

        from sklearn.decomposition import NMF
        model = NMF(n_components=20)

        train = pd.DataFrame(list(user_item_train_df.all_ratings()), columns=['user', 'item', 'rating'])        

        def ConvertToDense(X, y, shape):  # from R=(X,y), in sparse format 
            row  = X[:,0]
            col  = X[:,1]
            data = y
            matrix_sparse = sparse.csr_matrix((data,(row,col)), shape=(shape[0]+1,shape[1]+1))  # sparse matrix in compressed format (CSR)
            R = matrix_sparse.todense()   # convert sparse matrix to dense matrix, same as: matrix_sparse.A
            R = R[1:,1:]                  # removing the "Python starts at 0" offset
            R = np.asarray(R)             # convert matrix object to ndarray object
            return R
        
        X = train[['user', 'item']].values
        y = train['rating'].values

        n_users = max(train['user'].unique())+1
        n_items = max(train['item'].unique())+1
        R_shape = (n_users, n_items)

        Ra = ConvertToDense(X, y, R_shape)

        model.fit(Ra.T)                     
        self.W = model.transform(Ra.T)  
        self.H = model.components_.T

        self.predictions = np.dot(self.H, self.W.T)


    def test(self, data):
        user_id = int(data[0][0])
        
        predictions = sorted(
            [(ind, i) for ind, i in enumerate(self.predictions[user_id])],
            key=lambda x: x[1],
            reverse=True
        )

        return predictions