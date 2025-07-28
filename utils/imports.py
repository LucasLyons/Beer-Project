import scipy as sp
import scipy.stats as stats
import powerlaw as pl
import kagglehub
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import duckdb as db
import recbole as rb
import sklearn as sk
import itertools
import pickle
import surprise
from scipy.sparse import coo_matrix, csr_matrix, save_npz, load_npz
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity