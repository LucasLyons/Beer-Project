import seaborn as sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict
from surprise import Dataset, SVD, Reader

def get_beer_data(beer_data, beer_ids, item_encoder=None, sort = False):
    if item_encoder:
        beer_ids = item_encoder.inverse_transform(beer_ids)
    results = beer_data[
        beer_data['beer_beerid'].isin(beer_ids)
        ].groupby([
        'beer_beerid', 'beer_name', 'brewery_name', 'beer_style'
        ], group_keys=False, as_index=False
        ).agg(
            {'review_overall': ['mean', 'count', 'std'],'beer_abv': ['mean']}
            )
    if sort:
        result = results.sort_values(
            by=('review_overall', 'count'), ascending=False
            ).set_index("beer_beerid")
    return results

def plot_heatmap(grid_search, values, ax=None):
    if ax is not None:
        ax.clear()
        pivot = grid_search.copy().pivot(
            index="k", columns="reg", values=values)
        sb.heatmap(
            pivot, annot=True, fmt=".4f", cmap='rocket', ax=ax, cbar=False
            ).invert_yaxis()
    else:
        pivot = grid_search.copy().pivot(index="k", columns="reg", values=values)
        sb.heatmap(pivot, annot=True, fmt=".4f", cmap='rocket').invert_yaxis()

def plot_comparison_heatmap(data_1, data_2, values):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # plot heatmaps
    hm1 = plot_heatmap(data_1, values, ax=axes[0])
    hm2 = plot_heatmap(data_2, values, ax=axes[1])

    # Create a common color scale (vmin, vmax) for both
    vmin = min(data_1[values].min(), data_2[values].min())
    vmax = max(data_1[values].max(), data_2[values].max())

    # Create a single colorbar
    # Use the image from one of the heatmaps to generate the colorbar
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # [left, bottom, width, height]
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="rocket", norm=norm)
    sm.set_array([])
    n_ticks = 4
    tick_values = np.linspace(vmin, vmax, n_ticks)
    fig.colorbar(sm, cax=cbar_ax, ticks=tick_values)
    return fig, axes


def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

    from surprise import SVD

def top_n_coverage(model, trainset, N=10):
    # Extract learned latent matrices
    P = model.pu        # shape: n_users × n_factors
    Q = model.qi.T      # shape: n_factors × n_items
    preds = P @ Q

    # mask known interactions
    for uid in range(trainset.n_users):
        for iid, _ in trainset.ur[uid]:
            preds[uid, iid] = -np.inf

    # Top-N indices
    top_n_iids = np.argpartition(-preds, N, axis=1)[:, :N]
    item_coverage = len(np.unique(top_n_iids)) / trainset.n_items
    return item_coverage 

def round_half(x):
    return round(x * 2) / 2

def safe_len(obj):
    try:
        return len(obj)
    except TypeError:
        return 1
    
def surprise_get_top_N(algo, N=10):
    import numpy as np
    from collections import defaultdict

    # Extract latent factors
    U = algo.pu               # user latent factors (n_users, n_factors)
    V = algo.qi               # item latent factors (n_items, n_factors)
    bu = algo.bu              # user biases (n_users,)
    bi = algo.bi              # item biases (n_items,)
    global_mean = algo.trainset.global_mean

    # Map inner IDs to raw IDs
    uid_map = {uid: algo.trainset.to_raw_uid(uid) for uid in algo.trainset.all_users()}
    iid_map = {iid: algo.trainset.to_raw_iid(iid) for iid in algo.trainset.all_items()}

    # Set of all item inner IDs
    all_item_inner_ids = set(algo.trainset.all_items())

    top_n = defaultdict(list)

    for uid_inner in algo.trainset.all_users():
        seen_iids_inner = set(j for (j, _) in algo.trainset.ur[uid_inner])
        unseen_iids_inner = np.array(list(all_item_inner_ids - seen_iids_inner))

        if len(unseen_iids_inner) == 0:
            continue

        # Vectorized prediction: u ⋅ vᵀ + bu + bi + μ
        user_vec = U[uid_inner]                    # shape: (n_factors,)
        unseen_item_vecs = V[unseen_iids_inner]    # shape: (n_unseen, n_factors)
        scores = unseen_item_vecs @ user_vec       # dot product
        scores += bu[uid_inner] + bi[unseen_iids_inner] + global_mean

        # Top-n indices
        topn_indices = np.argpartition(scores, -N)[-N:]
        topn_sorted_indices = topn_indices[np.argsort(scores[topn_indices])[::-1]]
        topn_iids_inner = unseen_iids_inner[topn_sorted_indices]

        # Convert back to raw IDs and store
        uid_raw = uid_map[uid_inner]
        top_n[uid_raw] = [(iid_map[iid], scores[topn_sorted_indices[i]]) for i, iid in enumerate(topn_iids_inner)]
        return top_n