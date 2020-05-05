import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

'''
# Reading user file
u_cols = ['user_id', 'age', 'occuapation', ' zip_code']
users = pd.read_csv('ml-100k/u.user', sep = '|', names = u_cols, encoding = 'latin-1')
n_users = users.shape[0]
#print('Number of users: ',n_users)
#print(users.head())

# Reading rating file
r_cols = ['user_id', 'moive_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('ml-100k/ua.base', sep = '\t', names = r_cols, encoding = 'latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep = '\t', names = r_cols, encoding = 'latin-1')
rate_train = ratings_base.values
rate_test = ratings_test.values
#print('Number of training rates: ', rate_train.shape[0])
#print('Number of test rates: ', rate_test.shape[0])

# Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror', 'Musical',
 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')
n_items = items.shape[0]
#print ('Number of items:', n_items)
#print(items.head())
'''

# Class CF
class CF(object):

    def __init__(self, Y_data, k, dist_func = cosine_similarity, uuCF = 1):
        self.uuCF = uuCF #user-based (1) or item-based (0)
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]] # convert between user_id and item_id 
        self.k = k # number of neigbor points
        self.dist_func = dist_func
        self.Ybar_data = None
        self.n_users = int(np.max(self.Y_data[:, 0])) +1 # id starts from 0
        self.n_items = int(np.max(self.Y_data[:, 1])) +1
    
    def add(self, new_data):
        self.Y_data = np.concatenate((self.Y_data, new_data), axis = 0)
    
    def normalize_Y(self):
        users = self.Y_data[:, 0] # all users - fisrt col of the Y_data
        self.Ybar_data = self.Y_data.copy() 
        self.mu = np.zeros((self.n_users,))
        for n in range(self.n_users):
            # row indices of rating done by user n
            ids = np.where(users == n)[0].astype(np.int32)
            # indices of all ratings associated with user n
            item_ids = self.Y_data[ids, 1]
            # and the corresponding ratings
            ratings = self.Y_data[ids, 2]
            # take mean
            m = np.mean(ratings)
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            self.mu[n] = m
            # normalize
            self.Ybar_data[ids, 2] = ratings - self.mu[n]
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
            (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()

    def similarity(self):
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)
    
    def refresh(self):
        self.normalize_Y()
        self.similarity()

    def fit(self):
        self.refresh()
    
    def __pred(self, u, i, normalized = 1):
        # step 1: find all users who rated i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        # step 2:
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        # step 3: find similarity the current user and others who rated i
        sim = self.S[u, users_rated_i]
        # step 4: find the k most similarity users
        a = np.argsort(sim)[- self.k:]
        # and the correspongding similarity levels
        nearest_s = sim[a]
        r = self.Ybar[i, users_rated_i[a]]
        if normalized:
            return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8) # to avoid dividing by 0
        return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8) + self.mu[u]
    
    def pred(self, u, i, normalized = 1):
        if self.uuCF: return self.__pred(u, i, normalized)
        return self.__pred(i, u, normalized)
    
    def recommend(self, u, normalized = 1):
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()
        recommended_items = []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                if rating > 0:
                    recommended_items.append(i)
        return recommended_items

    def print_recommendation(self):
        print('Recommendation: ')
        for u in range(self.n_users):
            recommended_items = self.recommend(u)
            if self.uuCF:
                print('   Recommend items(s): ', recommended_items, ' to user', u)
            else:
                print('   Recommend item ', u, ' to user(s): ', recommended_items)
'''
# Data file Example
r_cols = ['user_id', 'item_id', 'rating']
ratings = pd.read_csv('ex.dat', sep = ' ', names = r_cols, encoding = 'latin-1')
Y_data = ratings.values

# User-based CF
rs = CF(Y_data, k = 2, uuCF = 1)
rs.fit()
rs.print_recommendation()

# Item-based CF
rs = CF(Y_data, k = 2, uuCF = 0)
rs.fit()
rs.print_recommendation()
'''

# Load data MovieLens 100k
r_cols = ['user_id', 'moive_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('ml-100k/ub.base', sep = '\t', names = r_cols, encoding = 'latin-1')
ratings_test = pd.read_csv('ml-100k/ub.test', sep = '\t', names = r_cols, encoding = 'latin-1')

rate_train = ratings_base.values
rate_test = ratings_test.values

rate_train[:, :2] -= 1
rate_test[:, :2] -= 1

# User-based CF
rs = CF(rate_train, k = 30, uuCF = 1)
rs.fit()
n_tests = rate_test.shape[0]
SE = 0 # squared error
for n in range(n_tests):
    pred = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized = 0)
    SE += (pred - rate_test[n, 2])**2 
RMSE = np.sqrt(SE/n_tests)
print ('User-based CF, RMSE =', RMSE)

# Item-based CF
rs = CF(rate_train, k = 30, uuCF = 0)
rs.fit()
n_tests = rate_test.shape[0]
SE = 0 # squared error
for n in range(n_tests):
    pred = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized = 0)
    SE += (pred - rate_test[n, 2])**2 
RMSE = np.sqrt(SE/n_tests)
print ('Item-based CF, RMSE =', RMSE)
