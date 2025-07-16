def predict_top_N(user, train, similarity, mean_scores, k=10, N=5):
    
    """This function will predict the top-N items for a user based on k-nn user-based collaborative filtering.

    Parameters:
    -user: user index for whom to predict ratings
    -item: item index for which to predict ratings
    -train: training data in sparse matrix format
    -similarity: similarity matrix in sparse format
    -mean_scores: mean scores for each user
    -k: number of nearest neighbouts to consider
    
    Returns: top-N predictions for user"""
    
    # save number of items
    n_items = train.shape[1]
    # get unrated items
    unrated = list(set(range(n_items)) - set(train[0].indices))

    preds = []    
    for item in unrated:
        # predict rating for each unrated item
        pred = predict(user, item, train, similarity, mean_scores, k, clipped=False)
        preds.append((item, pred))

    # get top-N predictions
    top_N = sorted(preds, key=lambda x: x[1])[-N:]

    return [i[0] for i in top_N] # return top-N predictions