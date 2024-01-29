# %% [markdown]
# # Generate Topic Data

# %%
import sys
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
from numpy.linalg import norm, svd, solve, qr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
import networkx as nx

import scipy
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment

import cvxpy as cp
from cvxpy import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.problems.problem import Problem

sys.path.append(os.path.join(os.getcwd(), 'SpLSI/pycvxcluster/src/'))
import pycvxcluster.pycvxclt

from SpLSI import generate_topic_model as gen_model
from SpLSI.utils import *
from SpLSI.spatialSVD import *
from SpLSI import splsi
#from netgraph import Graph
#from netgraph import get_sugiyama_layout
from sklearn.model_selection import ParameterGrid
import time
import pandas as pd

import multiprocessing 
from PIL import Image

def run(params):
    print(f'Running with params: {params}')
    wd = os.getcwd()
    N = params['N'] # doc length
    n = params['n'] # number of nodes
    p = params['p'] # vocab size
    seed = params['seed']
    K = 3 # number of topics
    r = 0.05 # heterogeneity parameter
    m = 5 # number of neighbors to be considered in weights
    phi = 0.1 # weight parameter

    save_dir = os.path.join(wd, 'results', f'seed_{seed}_N_{N}_n_{n}_p_{p}_K_{K}_r_{r}_m_{m}_phi_{phi}')
    os.makedirs(save_dir, exist_ok=True)
    timing = pd.DataFrame()
    timing['seed'] = [seed]
    timing['N'] = [N]
    timing['n'] = [n]
    timing['p'] = [p]
    timing['K'] = [K]
    timing['r'] = [r]
    timing['m'] = [m]
    timing['phi'] = [phi]

    np.random.seed(seed)

    df, W, A, D = gen_model.generate_data(N, n, p , K, r)
    weights = gen_model.generate_weights(df, K, m, phi)
    G, mst, path = generate_mst(df, weights, n)
    srn, fold1, fold2 = get_folds(mst, path, n, plot_tree=False)
    folds = {0:fold1, 1:fold2}


    gen_model.plot_scatter(df)
    plt.savefig(os.path.join(save_dir, 'topic_data.png'), dpi=300)
    # %% [markdown]
    # # MST Folds / Matrix Denoising via Iterative Convex Clustering

    # %%
    colors = gen_model.get_colors(df)
    # Plot the random graph
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=False, node_size=10, node_color=colors, edge_color='gray', alpha=0.6)
    

    # Plot the minimum spanning tree in red
    nx.draw(mst, pos, with_labels=False, node_size=10, node_color=colors, edge_color='r', alpha=1)

    plt.savefig(os.path.join(save_dir, 'mst.png'), dpi=300)

    # %% [markdown]
    # # Spatial pLSI

    # %% [markdown]
    # #### Test SpatialSVD

    # %% [markdown]
    # ### Spatial SVD

    # %%
    spl = splsi.SpLSI(lamb_start = 0.0001,
                step_size = 1.2,
                grid_len = 50,
                step = 'two-step')
    spl_start = time.perf_counter()
    spl.fit(D, K, df, weights)
    spl_end = time.perf_counter()
    timing['spatial_svd_time'] = [spl_end - spl_start]

    # %%
    W_hat = spl.W_hat
    P = get_component_mapping(W_hat.T, W)
    W_hat = W_hat @ P
    err = get_F_err(W_hat, W)
    acc = get_accuracy(df, n, W_hat)
    print(f"Error is {err}.")
    timing['spatial_svd_err'] = [err]
    print(f"Accuracy is {acc}.")
    timing['spatial_svd_acc'] = [acc]

    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
    for j, ax in enumerate(axes1):
        w = W_hat[:, j]
        df[f'w{j+1}'] = w
        sns.scatterplot(x='x', y='y', hue=f'w{j+1}', data=df, palette='viridis', ax=ax)
        ax.set_title(f'Plot {j+1}')
    fig1.suptitle('Spatial SVD')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spatial_svd.png'), dpi=300)

    # %% [markdown]
    # ### Vanilla SVD

    # %%
    spl_v = splsi.SpLSI(lamb_start = 0.0001,
                step_size = 1.2,
                grid_len = 50,
                method = "nonspatial")
    spl_v_start = time.perf_counter()
    spl_v.fit(D, K, df, weights)
    spl_v_end = time.perf_counter()
    timing['vanilla_svd_time'] = [spl_v_end - spl_v_start]

    # %%
    W_hat = spl_v.W_hat
    P = get_component_mapping(W_hat.T, W)
    W_hat = W_hat @ P
    err = get_F_err(W_hat, W)
    acc = get_accuracy(df, n, W_hat)
    print(f"Error is {err}.")
    timing['vanilla_svd_err'] = [err]
    print(f"Accuracy is {acc}.")
    timing['vanilla_svd_acc'] = [acc]

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    for j, ax in enumerate(axes2):
        w = W_hat[:, j]
        df[f'w{j+1}'] = w
        sns.scatterplot(x='x', y='y', hue=f'w{j+1}', data=df, palette='viridis', ax=ax)
        ax.set_title(f'Plot {j+1}')
    fig2.suptitle('Vanilla SVD')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vanilla_svd.png'), dpi=300)

    # %% [markdown]
    # ### Ground Truth

    # %%
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
    for j, ax in enumerate(axes3):
        w = W[j,:]
        df[f'w{j+1}'] = w
        sns.scatterplot(x='x', y='y', hue=f'w{j+1}', data=df, palette='viridis', ax=ax)
        ax.set_title(f'Plot {j+1}')
    fig3.suptitle('Ground Truth')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ground_truth.png'), dpi=300)

    #combine spatial, vanilla, and ground truth pngs
    im1 = Image.open(os.path.join(save_dir, 'spatial_svd.png'))
    im2 = Image.open(os.path.join(save_dir, 'vanilla_svd.png'))
    im3 = Image.open(os.path.join(save_dir, 'ground_truth.png'))
    im_list = [im2, im1, im3]
    imgs_comb = np.vstack( [np.asarray( i ) for i in im_list ] )
    imgs_comb = Image.fromarray( imgs_comb)
    imgs_comb.save( os.path.join(save_dir, 'comparison.png') )

    timing.to_csv(os.path.join(save_dir, 'timing.csv'), index=False)
    print(f'Finished with params: {params}')

if __name__ == "__main__":
    param_grid = {'N':[10, 30, 50, 100, 200],
              'n':[1000],
              'p':[10, 30, 50],
              'seed':[i for i in range(1, 101)]}
    # Create a pool of processes
    pool = multiprocessing.Pool()

    # Map the helper function to the parameter grid using multiple processes
    pool.map(run, ParameterGrid(param_grid))

    # Close the pool of processes
    pool.close()
    pool.join()
    print('Aggregating results')
    #aggregate csvs
    wd = os.getcwd()
    results_dir = os.path.join(wd, 'results')
    csvs = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.csv') and file != 'results.csv':
                csvs.append(os.path.join(root, file))
    dfs = []
    for csv in csvs:
        dfs.append(pd.read_csv(csv))
    df = pd.concat(dfs)
    df.to_csv(os.path.join(results_dir, 'results.csv'), index=False)
    print('Done')