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
        mu = train.data.mean()
        self.mu = mu # update attribute
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
        RMSE_past = 0.001
        
        #loop until relative RMSE improvement threshold is < epsilon or patience runs out
        for t in range(self.patience):
            # randomize order for SGD
            np.random.shuffle(interactions)
            # loop over all interactions and update params
            for u, i, rating in interactions:
                self.__update(mu, u, i, rating)
            # get RMSE
            RMSE = self.__get_val_RMSE(mu, self.B_u, self.B_i, self.P, self.Q, validation)
            # calculate improvement threshold
            threshold = np.abs(RMSE-RMSE_past) / RMSE_past
            t += 1
            # break if RMSE stops improving
            if threshold < self.epsilon:
                self.RMSE = RMSE
                print(f'Stopped after {t} iterations')
                print(f'Final RMSE is: {RMSE} with {self.k} latent factors, {self.lr} learning rate, {self.reg} reg. parameter')
                break
            # update RMSE
            RMSE_past = RMSE
            if verbose == True:
                print(f'Iteration: {t}')
                print(f'current validation RMSE: {RMSE}')
        return
    
    def __update(self, mu, u, i, rating):
        """
        Update SVD model parameters in a pass of SGD.
        Args:
            -mu: global mean of ratings
            -u: user index
            -i: item index
            -rating: user u's rating of item i
        """
        #predict rating
        e = rating - (mu + self.B_u[u] + self.B_i[i] + self.P[u] @ self.Q[i])
        #make parameter updates
        self.B_u[u] += self.lr * (e-self.reg*self.B_u[u])
        self.B_i[i] += self.lr * (e-self.reg*self.B_i[i])
        self.Q[i] += self.lr * (e*self.P[u]-self.reg*self.Q[i])
        self.P[u] += self.lr * (e*self.Q[i]-self.reg*self.P[u])

    def __get_val_RMSE(self, mu, B_u, B_i, P, Q, validation):
        """
        Generate predictions on validation data and return RMSE.
        Args:
            -mu: global mean of ratings
            -B_u: user biases
            -B_i: item biases
            -P: user factors
            -Q: item factors
        """
        # get values
        user_idx = validation['user_idx'].values
        item_idx = validation['item_idx'].values
        ratings = validation['review_overall'].values
        # get factor scores
        factor_scores = np.sum(np.multiply(
            P[user_idx], # user factors
            Q[item_idx] # item factors
        ), axis = 1)

        # generate predictions
        preds = mu + B_u[user_idx] + B_i[item_idx] + factor_scores
        # calculate error
        errors = ratings - preds
        # calculate RMSE
        RMSE = np.sqrt(np.mean(errors**2))
        return RMSE
    
    def coverage(self):
        # get user and item bias vectors
        B_u_row = self.B_u.reshape(self.n_users,1)
        B_i_col = self.B_i.reshape(1,self.n_items)
        # generate matrix of user-item predictions
        preds = self.mu + B_u_row + B_i_col + self.P @ self.Q.T
        