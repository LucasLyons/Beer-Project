import scipy as sp
import scipy.stats as stats
import pandas as pd
import numpy as np
from utils.helpers import safe_len
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ManualSVD():

    def __init__(self, k=50, bias_lr=0.005, latent_lr = 0.005, reg_bias=0.2,
                  reg_latent=0.02, patience=20, epsilon=10**(-3)):
        # initialize hyperparameters
        self.k = k
        self.bias_lr = bias_lr
        self.latent_lr = latent_lr
        self.reg_bias = reg_bias
        self.reg_latent = reg_latent
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
        self.train = None # sparse matrix (memory efficient)
        self.random_seed = 420
        self.item_freq = None
        self.user_freq = None
        # if using validation set
        self.val_RMSE = 0
        self.val_RMSE_clipped = 0
        self.val_MAE = 0

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
    
    def fit(self, train, validation=None, verbose=True):
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
        interactions = self.__fit_initialize(train)
        
        if validation is not None:
            self.__validation_compute(validation, interactions, verbose)
        else: 
            for t in range(1, self.patience+1):
                for u, i, rating in interactions:
                    self.__update(u, i, rating)
            print(f'Params: {self.k} latent factors')
            print(f'Stopped after {t} iterations')
        
        return
    
    def __fit_initialize(self, train):
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
        from collections import Counter
        self.item_freq = np.bincount(self.train.col)  # Fast for COO
        self.user_freq = np.bincount(self.train.row)
        return interactions
    
    def __validation_compute(self, validation, interactions, verbose=False):
        # initialize RMSE counter
        RMSE_past = None
        #loop until relative RMSE improvement threshold is < epsilon or patience runs out
        for t in range(1, self.patience+1):
            # randomize order for SGD
            np.random.shuffle(interactions)
            # loop over all interactions and update params
            for u, i, rating in interactions:
                self.__update(u, i, rating)
            # get RMSE on validation set
            preds = self.predict_validation(validation, clipped=False)
            eval = self.accuracy(preds) # returns RMSE, MAE
            RMSE = eval[0]
            # calculate optional stopping after first iteration
            if t > 1:
                # calculate improvement threshold
                threshold = np.abs(RMSE-RMSE_past) / RMSE_past
                # break if RMSE stops improving or patience runs out
                if (threshold < self.epsilon) or (t == self.patience):
                    # get RMSE
                    self.val_RMSE = RMSE
                    # save predictions errors
                    self.val_MAE = eval[1] #MAE
                    self.val_RMSE_clipped = eval[0] # and clipped RMSE
                    print(f'Final validation RMSE is: {self.val_RMSE}\n')
                    print(f'Stopped after {t} iterations')
                    break
            # update RMSE
            RMSE_past = RMSE
            if verbose:
                print(f'Iteration: {t}')
                print(f'current validation RMSE: {RMSE}')
    
    def __update(self, u, i, rating):
        """
        Update SVD model parameters in a pass of SGD.
        Args:
            -u: user index
            -i: item index
            -rating: user u's rating of item i
        """
        n_i = self.item_freq[i]
        n_u = self.user_freq[u]
        #predict rating
        e = (rating - (self.mu + self.B_u[u] + self.B_i[i] + np.dot(self.P[u], self.Q[i])))
        #make parameter updates
        self.B_u[u] += self.bias_lr * (e-(self.reg_bias/np.sqrt(n_u))*self.B_u[u])
        self.B_i[i] += self.bias_lr * (e-(self.reg_bias/np.sqrt(n_i))*self.B_i[i]) # regularize bias terms
        self.Q[i] += self.latent_lr * (e*self.P[u]-self.reg_latent*self.Q[i])
        self.P[u] += self.latent_lr * (e*self.Q[i]-self.reg_latent*self.P[u])

    def predict_validation(self, validation, clipped=True):
        """
        Generate predictions on validation data and return.
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
        if clipped:
            preds = np.round(preds * 2) / 2 # round to nearest 0.5
            preds = np.clip(preds, 0, 5) # clip to [0,5]
        return (ratings, preds)
    
    def predict(self, users, items, clipped=False):
        """
        Predict the rating of users for items.
        Args:
            -users: users whose ratings will be predicted
            -items: predict the users' rating of these items
            -clipped: if True, round to nearest 0.5 and clip to [0,5]
        Returns:
            prediction: |users| x |items| matrix of predicted ratings
        """
        u, i = safe_len(users), safe_len(items) # |users|, |items|
        prediction = self.mu + self.B_u[users].reshape(u,1) + self.B_i[items].reshape(1,i) \
            + (self.P[users,:] @ self.Q[items,:].T) # dim |users| x |items|
        if clipped == False:
            return prediction
        clipped_prediction = np.round(prediction * 2) / 2 # round to nearest 0.5
        clipped_prediction = np.clip(clipped_prediction, 0, 5) # clip to [0,5]
        return clipped_prediction
    
    def accuracy(self, predict_validation, verbose=False):
        """
        Computes the RMSE and MAE of the model on the validation set.
        Args:
            -predict_validation: validation set with predictions
        Returns:
            RMSE, MAE: root mean squared error and mean absolute error
        """
        # get values
        ratings, preds = predict_validation
        # calculate errors
        RMSE = np.sqrt(mean_squared_error(ratings, preds))
        MAE = mean_absolute_error(ratings, preds)
        if verbose:
            print(f'RMSE: {RMSE}, MAE: {MAE}')
        return RMSE, MAE
    
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
    
    def get_top_N(self, biases=True, N=10, **kwargs):
        """
        Computes, the top-N unseen items, as ranked by user-item factor interactions.
        Args:
            -N: the number of unseen items recommended
        """
        a = kwargs.get('a', 1) 
        # generate matrix of user-item predictions
        preds = (self.P @ self.Q.T)
        if biases:
            preds += (self.B_u.reshape(-1, 1) + self.B_i.reshape(1, -1)) * a
        # find "seen" items (previously rated)
        user_ids, item_ids = self.train.row, self.train.col
        # mask seen items
        masked_preds = preds.copy()
        masked_preds[user_ids, item_ids] =  - np.inf
        top_n = np.argpartition(masked_preds, -N, axis=1)[:,-N:]
        # Step 1: Get the top-N *unsorted* item scores
        top_n_scores = np.take_along_axis(masked_preds, top_n, axis=1)  # shape: (num_users, N)

        # Step 2: Get sort order (descending) for each row's top-N scores
        sort_order = np.argsort(top_n_scores, axis=1)[:, ::-1]  # shape: (num_users, N)

        # Step 3: Reorder the top-N item indices by their score
        top_n_sorted = np.take_along_axis(top_n, sort_order, axis=1)  # shape: (num_users, N)
        return top_n_sorted
    
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

        