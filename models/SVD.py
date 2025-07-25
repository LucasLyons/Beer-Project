import scipy as sp
import scipy.stats as stats
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

class SVD():

    def __init__(self, k=50, lr=0.005, reg=0.02, patience=50, epsilon=10**(-3)):
        # initialize hyperparameters
        self.k = k
        self.lr = lr
        self.reg = reg
        self.patience = patience
        self.epsilon = epsilon
        # model parameters will be set during fit
        self.B_u = None
        self.B_i = None
        self.P = None
        self.Q = None
        self.mu = None
        # training set
        self.n_users = None
        self.n_items = None
        self.RMSE = 0
        self.RMSE_clipped = 0
        self.train = None # sparse matrix (memory efficient)
        self.random_seed = 420

    def set_params(self, B_u = None, B_i = None, P = None, Q = None, mu = None):
        """
        Set model parameters manually.
        Args:
            -B_u: user biases
            -B_i: item biases
            -P: user factors
            -Q: item factors
            -mu: global mean of ratings
        """
        if B_u is not None:
            self.B_u = B_u
        if B_i is not None:
            self.B_i = B_i
        if P is not None:
            self.P = P
        if Q is not None:
            self.Q = Q
        if mu is not None:
            self.mu = mu
    
    def fit(self, train, validation, verbose=True):
        """
        Computes the SVD model parameters using stochastic gradient descent.
        Ends after patience # of epochs have passed or the relative RMSE improvement threshold is < epsilon

        Args:
            train: sparse matrix of training data
            validation: sparse matrix of validation data
            k: number of latent factors
            lr: learning rate for SGD
            reg: regularization hyperparameter for learned parameters
            patience: maximum number of epochs to run SGD
            epsilon: Relative RMSE improvement threshold cutoff
        """
        # use random seed
        np.random.seed(self.random_seed)
        # make sure train is in COO format
        train = train.tocoo()
        self.train = train
        # count users and items
        self.n_users = train.shape[0]
        self.n_items = train.shape[1]
        # get global mean rating
        self.mu = train.data.mean()
        # initialize biases
        self.B_u = np.zeros(self.n_users)
        self.B_i = np.zeros(self.n_items)
        # initialize factors
        self.P = np.random.normal(loc=0.0, scale=0.1, size=(self.n_users, self.k))
        self.Q = np.random.normal(loc=0.0, scale=0.1, size=(self.n_items, self.k))
        # store all interactions
        interactions = list(zip(
            train.row, #get rows
            train.col, #get cols
            train.data #get ratings
        ))
        # initialize RMSE counter
        RMSE_past = None
        
        #loop until relative RMSE improvement threshold is < epsilon or patience runs out
        for t in range(1, self.patience+1):
            # randomize order for SGD
            np.random.shuffle(interactions)
            # loop over all interactions and update params
            for u, i, rating in interactions:
                self.__update(u, i, rating)
            # get RMSE
            RMSE = self.__get_val_RMSE(validation)
            # calculate improvement threshold
            if t > 1:
                threshold = np.abs(RMSE-RMSE_past) / RMSE_past
                # break if RMSE stops improving
                if threshold < self.epsilon:
                    # get RMSE
                    self.RMSE = RMSE
                    # and clipped RMSE
                    self.RMSE_clipped = self.get_clipped_pred_RMSE(validation)
                    print(f'Stopped after {t} iterations')
                    print(f'Final RMSE is: {self.RMSE} (clipped prediction RMSE is {self.RMSE_clipped}) \n' 
                          f'Params: {self.k} latent factors, '
                          f'{self.lr} learning rate, {self.reg} reg. parameter')
                    break
            # update RMSE
            RMSE_past = RMSE
            if verbose == True:
                print(f'Iteration: {t}')
                print(f'current validation RMSE: {RMSE}')
        return
    
    def __update(self, u, i, rating):
        """
        Update SVD model parameters in a pass of SGD.
        Args:
            -u: user index
            -i: item index
            -rating: user u's rating of item i
        """
        #predict rating
        e = rating - (self.mu + self.B_u[u] + self.B_i[i] + self.P[u] @ self.Q[i])
        #make parameter updates
        self.B_u[u] += self.lr * (e-self.reg*self.B_u[u])
        self.B_i[i] += self.lr * (e-self.reg*self.B_i[i])
        self.Q[i] += self.lr * (e*self.P[u]-self.reg*self.Q[i])
        self.P[u] += self.lr * (e*self.Q[i]-self.reg*self.P[u])

    def __get_val_RMSE(self, validation):
        """
        Generate continuous predictions on validation data and return RMSE.
        Args:
            -validation: validation set consisting of user-item interactions
        """
        # get values
        user_idx = validation['user_idx'].values
        item_idx = validation['item_idx'].values
        ratings = validation['review_overall'].values
        # get factor scores
        factor_scores = np.sum(np.multiply(
            self.P[user_idx], # user factors
            self.Q[item_idx] # item factors
        ), axis = 1)

        # generate predictions
        preds = self.mu + self.B_u[user_idx] + self.B_i[item_idx] + factor_scores
        # calculate error
        errors = ratings - preds
        # calculate RMSE
        RMSE = np.sqrt(np.mean(errors**2))
        return RMSE
    
    def get_clipped_pred_RMSE(self, validation):
        """
        Generate continuous predictions on validation data and return RMSE.
        Args:
            -validation: validation set consisting of user-item interactions
        """
        # get values
        user_idx = validation['user_idx'].values
        item_idx = validation['item_idx'].values
        ratings = validation['review_overall'].values
        # get factor scores
        factor_scores = np.sum(np.multiply(
            self.P[user_idx], # user factors
            self.Q[item_idx] # item factors
        ), axis = 1)

        # generate predictions
        preds = self.mu + self.B_u[user_idx] + self.B_i[item_idx] + factor_scores
        preds = np.round(preds * 2) / 2 # round to nearest 0.5
        preds = np.clip(preds, 0, 5) # clip to [0,5]
        # calculate error
        errors = ratings - preds
        # calculate RMSE
        clipped_RMSE = np.sqrt(np.mean(errors**2))
        return clipped_RMSE
    
    def predict(self, user, item):
        """"
        Predict the rating of user u for item i. Prediction rounded to nearest 0.5 and clipped to [0,5].
        Args:
            -user: user whose rating will be predicted
            -item: predict the user's rating of this item
        """
        prediction = self.mu + self.B_u[user] + self.B_i[item] + self.P[user,:] @ self.Q[item,:]
        clipped_prediction = np.round(prediction * 2) / 2 # round to nearest 0.5
        clipped_prediction = np.clip(clipped_prediction, 0, 5) # clip to [0,5]
        return clipped_prediction
    
    def top_N_coverage(self, N=10):
        """
        Computes the coverage on the item catalog from the training set, 
        with top-N unseen items being recommended.
        Args:
            -N: the number of unseen items recommended
        """
        # generate matrix of user-item predictions
        preds = self.P @ self.Q.T
        # find "seen" items (previously rated)
        user_ids, item_ids = self.train.row, self.train.col
        # mask seen items
        masked_preds = preds.copy()
        masked_preds[user_ids, item_ids] =  - np.inf
        top_N = np.argpartition(-masked_preds, N, axis=1)[:, :N]

        # flatten and get unique items across all users
        unique_recommended_items = np.unique(top_N)
        coverage = round((len(unique_recommended_items) / preds.shape[1]),4)  # divide by total number of items
        return coverage
    
    def get_top_N(self, N=10):
        """
        Computes, the top-N unseen items, as ranked by user-item factor interactions.
        Args:
            -N: the number of unseen items recommended
        """
        # generate matrix of user-item predictions
        preds = self.P @ self.Q.T
        # find "seen" items (previously rated)
        user_ids, item_ids = self.train.row, self.train.col
        # mask seen items
        masked_preds = preds.copy()
        masked_preds[user_ids, item_ids] =  - np.inf
        top_N = np.argsort(-masked_preds, N, axis=1)[:, :N]

        return top_N
    
    def hit_rate_at_N(self, validation, N=100):
        """
        Computes the hit-rate, i.e. if the user's next-rated item is among the top N items recommended.
        Args:
            -validation: validation set
            -N: top N items are recommended
        """
        """
        Computes the coverage on the item catalog from the training set, 
        with top-N unseen items being recommended.
        Args:
            -N: the number of unseen items recommended
        """
        # generate matrix of user-item predictions
        preds = self.P @ self.Q.T
        # find "seen" items (previously rated)
        user_ids, item_ids = self.train.row, self.train.col
        # mask seen items
        masked_preds = preds.copy()
        masked_preds[user_ids, item_ids] =  - np.inf
        # get top N items
        top_N = np.argpartition(-masked_preds, N, axis=1)[:, :N]
        # calculate hits
        hits = (top_N[validation['user_idx'].values] == validation['item_idx'].values[:,np.newaxis] # check matches
                ).any(axis=1) # see if any of the TOp-N recommended were the validation set item
        hit_rate = np.mean(hits)
        return hit_rate

        