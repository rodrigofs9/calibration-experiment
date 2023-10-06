import numpy as np
from tqdm import trange
import pandas as pd
from scipy.sparse import csr_matrix
from source.metrics.metrics import Metrics
from scipy.spatial.distance import jensenshannon


import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np 
import pandas as pd 
import scipy.sparse as sp
from scipy.spatial.distance import jensenshannon

import torch.utils.data as data


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from tqdm import tqdm
import time




class BPRData(data.Dataset):
	def __init__(self, features, 
				num_item, train_mat=None, num_ng=0, is_training=None):
		super(BPRData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features = features
		self.num_item = num_item
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training

	def ng_sample(self):
		assert self.is_training, 'no need to sampling when testing'

		self.features_fill = []
		for x in self.features:
			u, i = x[0], x[1]
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while (u, j) in self.train_mat:
					j = np.random.randint(self.num_item)
				self.features_fill.append([u, i, j])

	def __len__(self):
		return self.num_ng * len(self.features) if \
				self.is_training else len(self.features)

	def __getitem__(self, idx):
		features = self.features_fill if \
				self.is_training else self.features

		user = features[idx][0]
		item_i = features[idx][1]
		item_j = features[idx][2] if \
				self.is_training else features[idx][1]
		return user, item_i, item_j 


class BPRM(nn.Module):
	def __init__(self, user_num, item_num, factor_num):
		super(BPRM, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors.
		"""		
		self.embed_user = nn.Embedding(user_num, factor_num)
		self.embed_item = nn.Embedding(item_num, factor_num)

		nn.init.normal_(self.embed_user.weight, std=0.01)
		nn.init.normal_(self.embed_item.weight, std=0.01)

	def forward(self, user, item_i, item_j):
		user = self.embed_user(user)
		item_i = self.embed_item(item_i)
		item_j = self.embed_item(item_j)

		prediction_i = (user * item_i).sum(dim=-1)
		prediction_j = (user * item_j).sum(dim=-1)
		return prediction_i, prediction_j

class BPR:
    def __init__(self, learning_rate = 0.01, n_factors = 15, n_iters = 10, 
                 batch_size = 1000, reg = 0.01, seed = 1234, verbose = True,
                 
                 
                 tradeoff=None, distribution_column=None, p_t_u_all_users=None, p_t_i_all_items=None,
                 dataset=None, movies_data=None, tipo='gs1', p_t_u_pop_all_users=None, p_t_i_pop_all_items=None,
                 tradeoff2 = None
                 
                 ):
        self.tradeoff = tradeoff
        self.distribution_column = distribution_column
        self.p_t_u_all_users = p_t_u_all_users
        self.p_t_i_all_items = p_t_i_all_items
        self.reg = reg
        self.lr = learning_rate
        self.lamda = reg
        self.seed = seed
        self.epochs = n_iters
        self.verbose = verbose
        self.n_iters = n_iters
        self.n_factors = n_factors
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.movies_data = movies_data
        self.tipo = tipo
        self.p_t_u_pop_all_users = p_t_u_pop_all_users
        self.p_t_i_pop_all_items = p_t_i_pop_all_items
        self.tradeoff2 = tradeoff2
        self.num_ng = 2
        
        self._prediction = None
        import time

    def test(self, data):
        s = time.time()
        user_id = int(data[0][0])

        # print("ShapeAll: ", self.embed_user.shape, self.embed_item.shape)
        preds = []
        user_emb = self.embed_user[user_id].reshape(1, -1)
        # print("tt", user_emb.shape)
        # print("Shape1: ", user_emb.shape)
        for item in range(self.item_num):
            items_emb = self.embed_item[item].reshape(-1, 1)
            # print(items_emb.shape)
            pred = user_emb.dot(items_emb)[0]
            # print("SHAPPEEE", pred.shape)
            preds.append(
                [item, pred]
            )
        
        

        return preds
 
    def fit(self, user_item_train_df):

        train_data = list(user_item_train_df.all_ratings())
        writer = SummaryWriter()

        user_num = 3828
        item_num = 2784

        self.item_num = item_num

        train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
        for x in train_data:
            train_mat[x[0], x[1]] = 1.0

        train_dataset = BPRData(
		    train_data, item_num, train_mat, self.num_ng, True)

        train_loader = data.DataLoader(train_dataset,
            batch_size=self.batch_size, shuffle=True, num_workers=4)

        self.model = BPRM(user_num, item_num, self.n_factors)
        optimizer = optim.SGD(
			self.model.parameters(), lr=self.lr, weight_decay=self.lamda)

        count, best_hr = 0, 0
        loss = 1
        genres = {'Crime', 'Western', 'Comedy', 'Thriller', 'Sci-Fi', 'Adventure', 'Fantasy', 'Animation', 'War', 'Action', 'Drama', 'Romance', 'Film-Noir', 'Mystery', "Children's", 'Musical', 'Documentary', 'Horror'}
        for epoch in tqdm(range(self.epochs), desc=f"Loss: {loss}"):
            self.model.train() 
            start_time = time.time()
            train_loader.dataset.ng_sample()

            for user, item_i, item_j in train_loader:
                loss = 0
                for uu, ii, jj in zip(user,item_i, item_j ):
                    i_dist = [self.p_t_i_all_items.get(int(ii.numpy()), {}).get(w, 0)+0.0001 for w in genres]
                    j_dist = [self.p_t_i_all_items.get(int(jj.numpy()), {}).get(w, 0)+0.0001 for w in genres]
                    u_dist = [self.p_t_u_all_users[int(uu.numpy())].get(w, 0)+0.0001 for w in genres]

                    div_ui = 1-jensenshannon(u_dist, i_dist)
                    div_uj = 1-jensenshannon(u_dist, j_dist)
                    a = 0
                    b = 0
                    if div_ui < div_uj:
                        a = 1
                        b = 0
                    else:
                        a = 0
                        b = 1


                    user = user
                    item_i = item_i
                    item_j = item_j

                    self.model.zero_grad()
                    prediction_i, prediction_j = self.model(uu, ii, jj)
                    loss = - (prediction_i*(1+a) - prediction_j*(1+b)).sigmoid().log().sum()
                loss.backward()
                optimizer.step()
                writer.add_scalar('data/loss', loss.item(), count)
                print(loss)
                count += 1
        print("TERMINOU FIT")
        self.model.eval()
        self.embed_item = self.model.embed_item.weight.detach().numpy().astype(np.float32)
        self.embed_user = self.model.embed_user.weight.detach().numpy().astype(np.float32)
        
        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()
        print("AGORA SIM TERMINOU FIT")