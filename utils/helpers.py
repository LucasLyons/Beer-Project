import seaborn as sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict
from surprise import Dataset, SVD, Reader

def get_beer_data(beer_data, beer_ids, item_encoder):
    results = beer_data[
        beer_data['beer_beerid'].isin(item_encoder.inverse_transform(beer_ids))
        ].groupby([
        'beer_beerid', 'beer_name', 'brewery_name', 'beer_style'
        ], group_keys=False, as_index=False
        ).agg(
            {'review_overall': ['mean', 'count', 'std'],'beer_abv': ['mean']}
            ).sort_values(by=('review_overall', 'count'), ascending=False).set_index("beer_beerid")
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