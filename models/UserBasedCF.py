import scipy as sp
import scipy.stats as stats
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

class UserBasedCF():
    def __init__(self, train, user_encoder, item_encoder):
        # initialize attributes
        self.train = train
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        # model parameters will be learned
        self.sim = None
        self.user_mean_scores = None
        
    def fit(self):
        # get sum of scores per user
        user_scores = self.train.sum(axis=1).A1
        # count number of user reviews
        user_counts = np.diff(self.train.indptr)
        # set mean vector
        self.user_mean_scores = user_scores / user_counts
        # get similarity matrix
        self.sim = cosine_similarity(self.train, dense_output=False)

    def predict(self, user, item, k=10, clipped=True):
        """
        Predict ratings for the user-item pair using k nearest neighbours, 
        rounded to the nearest .5 and capped in [0,5]. If item is unseen, default to global mean.
        
        Parameters:
        -user: user index for whom to predict ratings
        -item: item index for which to predict ratings
        -clipped: if True, rounds item scores to nearest 0.5 and clips to [0,5]
        -k: number of nearest-neighbours to use
        
        Returns: ordinal prediction for user-item pair
        """
        # find neighbours of user for item
        nbs = self.train[:,item].nonzero()[0]
        nbs = nbs[nbs != user] #exclude self
        if nbs.size == 0:
            # no neighbours, return mean score
            return self.user_mean_scores[user]
        
        # get ratings and mean-centre them
        ratings = self.train[nbs,item].toarray().flatten()
        ratings -= self.user_mean_scores[nbs]
        # set limit for k
        k = min(k, nbs.size)

        # get similarity scores
        sims = self.sim[user, :].toarray().flatten()
        # get similarity scores for neighbours
        sims = sims[nbs]
        # take k-nearest similarities
        sims = sims[np.argsort(sims)[-k:]]
        # get corresponding k-nearest ratings
        ratings = ratings[np.argsort(sims)[-k:]]
        # compute weighted average
        if np.sum(np.abs(sims)) == 0:
            return self.user_mean_scores[user]
        weighted_avg = np.dot(sims, ratings) / np.sum(np.abs(sims))
        # recenter
        weighted_avg += self.user_mean_scores[user]
        if clipped == False:
            return weighted_avg
        # round to nearest .5
        weighted_avg = np.round(weighted_avg * 2) / 2 
        # clip to [0,5]
        weighted_avg = np.clip(weighted_avg, 0, 5)
        # return prediction
        return weighted_avg
    
    def evaluate_error(self, validation, k=10):
        preds = []
        actuals = []

        for row in validation.itertuples(index=False):
            u = row.user_idx
            i = row.item_idx
            true_r = row.review_overall
            pred = self.predict(u, i, k)
            preds.append(pred)
            actuals.append(true_r)

            RMSE = np.sqrt(mean_squared_error(actuals, preds))
            MAE = mean_absolute_error(actuals,preds)

        return RMSE, MAE
    
    def predict_top_N(self, user, k=10, N=10):
        """
        Fast top-N prediction using user-based CF with vectorized matrix ops.

        Parameters:
        - user: target user index
        - N: number of top items to return

        Returns:
        - List of top-N item indices predicted for the user
        """
        # get top-k most similar users to target user
        user_sims = self.sim[user, :].toarray().flatten()
        topk_idx = np.argsort(user_sims)[-k:]
        topk_sims = user_sims[topk_idx]  # shape: (k,)

        # get their ratings and mean-center
        ratings = self.train[topk_idx, :].toarray()  # shape: (k, n_items)
        means = self.user_mean_scores[topk_idx][:, np.newaxis]
        ratings_centered = ratings - means  # shape: (k, n_items)

        # weighted sum of centered ratings
        numerator = topk_sims @ ratings_centered  # shape: (n_items,)
        denominator = np.sum(np.abs(topk_sims)) + 1e-8  # to avoid div by 0

        preds = self.user_mean_scores[user] + numerator / denominator  # shape: (n_items,)

        # mask out already rated items
        rated_items = self.train[user, :].nonzero()[1]
        preds[rated_items] = -np.inf  # exclude known ratings

        # return top-N items
        top_N_items = np.argsort(preds)[-N:][::-1]
        return top_N_items.tolist()
    
    def top_N_beers(self, user, k=3, N=5):
        """
        Predict top N beers with k-nn CF
        Args:
            -user: user idx
            -k: number of nearest neighbours to use in computation
            -N: return top-ranked beers up to rank N
        """
        preds = self.predict_top_N(user, k, N)
        # get item dict. mappings
        beer_ids = self.item_encoder.inverse_transform(preds)
        return beer_ids
    
    def coverage_at_N(self, k=3,N=5):
        recommended_beers = set()
        for user in range(self.train.shape[0]):
            preds = self.predict_top_N(user, k, N)
            # update the set of recommended beers
            recommended_beers.update(preds)
        # evaluate and return
        beers_recommended = len(recommended_beers)
        coverage = round(( beers_recommended / self.train.shape[1]) * 100,2)
        return (coverage, beers_recommended)