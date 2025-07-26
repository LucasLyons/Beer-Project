import seaborn as sb
import pandas as pd
import numpy as np

def get_beer_data(beer_data, beer_ids):
    results = beer_data[beer_data['beer_beerid'].isin(beer_ids)].groupby([
        'beer_beerid', 'beer_name', 'brewery_name', 'beer_style'], group_keys=False, as_index=False).agg(
            {'review_overall': ['mean', 'count', 'std'],
            'beer_abv': ['mean']}
        ).sort_values(by=('review_overall', 'count'), ascending=False).set_index("beer_beerid")
    return results

def plot_heatmap(grid_search, values):
    pivot = grid_search.copy().pivot(index="k", columns="reg", values=values)
    sb.heatmap(pivot, annot=True, fmt=".4f", cmap='rocket').invert_yaxis()