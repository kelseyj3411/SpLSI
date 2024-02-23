import sys
import os
import time
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

import pycvxcluster.pycvxcluster

from SpLSI import generate_topic_model as gen_model
from SpLSI.utils import *
from SpLSI.spatialSVD import *
from SpLSI import splsi
import utils.spatial_lda.model

from collections import defaultdict

def run_simul(nsim=50, N=100, n=1000, p=30, K=3, r=0.05, m=5, phi=0.1, lamb_start = 0.0001, step_size = 1.35, grid_len=30, start_seed=None):
    results = defaultdict(list)
    temp_save_loc = os.path.join(os.getcwd(), 'temp')
    #print(f"Running simulations for N={N}, n={n}, p={p}, K={K}, r={r}, m={m}, phi={phi}...")
    for trial in range(nsim):
        os.system(f'echo Running trial {trial}...')
        # Generate topic data and graph
        regens = 0
        if not (start_seed is None):
            np.random.seed(start_seed + trial)
        while True:
            try:
                coords_df, W, A, X = gen_model.generate_data(N, n, p, K, r)
                weights, edge_df = gen_model.generate_weights_edge(coords_df, m, phi)
                # Spatial SVD (two-step)
                start_time = time.time()
                model_splsi = splsi.SpLSI(lamb_start=lamb_start, step_size=step_size, grid_len=grid_len, step="two-step", verbose=0)
                model_splsi.fit(X, K, edge_df, weights)
                time_splsi = time.time() - start_time
                print(f"CV Lambda is {model_splsi.lambd}")
                # SLDA
                start_time = time.time()
                model_slda = utils.spatial_lda.model.run_simulation(X, K, coords_df)
                time_slda = time.time() - start_time
                break
            except Exception as e:
                print(f"Regenerating dataset due to error: {e}")
                regens += 1

        # Vanilla SVD
        start_time = time.time()
        model_v = splsi.SpLSI(lamb_start=lamb_start, step_size=step_size, grid_len=grid_len, method="nonspatial", verbose=0)
        model_v.fit(X, K, edge_df, weights)
        time_v = time.time() - start_time

        # Record [err, acc] for spl_v and spl_2
        P_v = get_component_mapping(model_v.W_hat.T, W)
        P_splsi = get_component_mapping(model_splsi.W_hat.T, W)
        P_slda = get_component_mapping(model_slda.topic_weights.values.T, W)

        W_hat_plsi = model_v.W_hat @ P_v
        W_hat_splsi = model_splsi.W_hat @ P_splsi
        W_hat_slda = model_slda.topic_weights.values @ P_slda
        err_acc_spl_v = [get_F_err(W_hat_plsi, W), get_accuracy(coords_df, n, W_hat_plsi)]
        err_acc_spl_splsi = [get_F_err(W_hat_splsi, W), get_accuracy(coords_df, n, W_hat_splsi)]
        err_acc_spl_slda = [get_F_err(W_hat_slda, W), get_accuracy(coords_df, n, W_hat_slda)]

        results['trial'].append(trial)
        results['seed'].append(start_seed + trial)
        results['regens'].append(regens)
        results['N'].append(N)
        results['n'].append(n)
        results['p'].append(p)
        results['K'].append(K)
        results['plsi_err'].append(err_acc_spl_v[0])
        results['plsi_acc'].append(err_acc_spl_v[1])
        results['splsi_err'].append(err_acc_spl_splsi[0])
        results['splsi_acc'].append(err_acc_spl_splsi[1])
        results['slda_err'].append(err_acc_spl_slda[0])
        results['slda_acc'].append(err_acc_spl_slda[1])
        results['plsi_time'].append(time_v)
        results['splsi_time'].append(time_splsi)
        results['slda_time'].append(time_slda)
        results['spatial_lambd'].append(model_splsi.lambd)
        results['splsi_iters'].append(model_splsi.used_iters)

        if not os.path.exists(temp_save_loc):
            os.makedirs(temp_save_loc)
        csv_loc = os.path.join(temp_save_loc, f'simul_N={N}_n={n}_K={K}_p={p}.csv')
        pd.DataFrame(results).tail(1).to_csv(csv_loc, index=False, mode='a', header = not os.path.exists(csv_loc))
    if os.path.exists(os.path.join(temp_save_loc, f'simul_N={N}_n={n}_K={K}_p={p}.csv')):
        os.remove(os.path.join(temp_save_loc, f'simul_N={N}_n={n}_K={K}_p={p}.csv'))
    results = pd.DataFrame(results)
    return results

def plot_fold_cv(lamb_start, step_size, grid_len, model, N):
    lambd_errs = model.lambd_errs
    lambd_grid = (lamb_start*np.power(step_size, np.arange(grid_len))).tolist()
    lambd_grid.insert(0, 1e-06)
    cv_1 = np.round(lambd_grid[np.argmin(lambd_errs['fold_errors'][0])], 5)
    cv_2 = np.round(lambd_grid[np.argmin(lambd_errs['fold_errors'][1])], 5)
    cv_final = model.lambd
    for j, fold_errs in lambd_errs['fold_errors'].items():
        plt.plot(lambd_grid, fold_errs, label=f'Fold {j}', marker = 'o')
        #plt.vlines(cv_1, 18.90, 18.25, color = "blue")
        #plt.vlines(cv_2, 18.90, 18.25, color = "orange")

    #plt.plot(lambd_grid, new_list, label='Final Errors', linestyle='--', linewidth=2)
    plt.xlabel('Lambda')
    plt.ylabel('Errors')
    plt.text(cv_1, lambd_errs['fold_errors'][0][0], cv_1, color='blue')
    plt.text(cv_1, lambd_errs['fold_errors'][1][0], cv_2, color='orange')
    plt.title(f'N={N}, Lambda CV = {cv_final}')
    plt.legend()



def plot_one_simul(N = 100, # doc length
              n = 1000, # number of nodes
              p = 30, # vocab size
              K = 3, # number of topics
              r = 0.05, # heterogeneity parameter
              m = 5, # number of neighbors to be considered in weights
              phi = 0.1, # weight parameter
              plot_what = False,
              plot_topic = False
              ):

    # Generate topic data and graph
    while True:
                try:
                    coords_df, W, A, X = gen_model.generate_data(N, n, p, K, r)
                    weights, edge_df = gen_model.generate_weights_edge(coords_df, m, phi)
                    break
                except Exception as e:
                    print(f"An error has occurred: {e}")
    row_ind = [('cell',i) for i in range(1000)]

    # Vanilla SVD
    start_time = time.time()
    model_v = splsi.SpLSI(lamb_start=0.0001, step_size=1.35, grid_len=30, method="nonspatial", verbose=0)
    model_v.fit(X, K, edge_df, weights)
    time_v = time.time() - start_time

    # Spatial SVD (two-step)
    start_time = time.time()
    model_splsi = splsi.SpLSI(lamb_start=0.0001, step_size=1.35, grid_len=30, step="two-step", verbose=0)
    model_splsi.fit(X, K, edge_df, weights)
    time_splsi = time.time() - start_time
    print(f"CV Lambda is {model_splsi.lambd}")

    # SLDA
    start_time = time.time()
    model_slda = utils.spatial_lda.model.run_simulation(X, K, coords_df)
    time_slda = time.time() - start_time

    # Record [err, acc] for spl_v and spl_2
    P_v = get_component_mapping(model_v.W_hat.T, W)
    P_splsi = get_component_mapping(model_splsi.W_hat.T, W)
    P_slda = get_component_mapping(model_slda.topic_weights.values.T, W)

    W_hat_v = model_v.W_hat @ P_v
    W_hat_splsi = model_splsi.W_hat @ P_splsi
    W_hat_slda = model_slda.topic_weights.values @ P_slda
    err_acc_spl_v = [get_F_err(W_hat_v, W), get_accuracy(coords_df, n, W_hat_v)]
    err_acc_spl_splsi = [get_F_err(W_hat_splsi, W), get_accuracy(coords_df, n, W_hat_splsi)]
    err_acc_spl_slda = [get_F_err(W_hat_slda, W), get_accuracy(coords_df, n, W_hat_slda)]

    names = ['Ground Truth Topics', "SPLSI Topics", "PLSI Topics", "SLDA Topics"]
    metrics = [err_acc_spl_splsi, err_acc_spl_v, err_acc_spl_slda]
    Whats = [W_hat_splsi, W_hat_v, W_hat_slda]
    times = [time_splsi, time_v, time_slda]

    if plot_what:
        print(f"Model: {names[0]}")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for j, ax in enumerate(axes):
            w = W[j, :]
            coords_df[f'w{j+1}'] = w
            sns.scatterplot(x='x', y='y', hue=f'w{j+1}', data=coords_df, palette='viridis', ax=ax)
            ax.set_title(f'True W {j+1}')
        plt.tight_layout()
        plt.show()

        for i, metric in enumerate(metrics):
            print(f"Model: {names[i+1]}")
            print(f"Error is {metric[0]}.")
            print(f"Accuracy is {metric[1]}.")
            print(f"Time is {times[i]}")
            What = Whats[i]

            # Plot the scatter plots for the model
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            for j, ax in enumerate(axes):
                w = What[:, j]
                coords_df[f'w{j+1}'] = w
                sns.scatterplot(x='x', y='y', hue=f'w{j+1}', data=coords_df, palette='viridis', ax=ax)
                ax.set_title(f'What {j+1}')
            plt.tight_layout()
            plt.show()
    
    if plot_topic:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        topic = np.argmax(W, axis=0)
        coords_df['grp'] = topic
        sns.scatterplot(x='x', y='y', hue='grp', data=coords_df, palette='viridis', ax=axes[0], legend=False)
        axes[0].set_title(names[0], fontsize=24, fontweight='bold')

        for i, metric in enumerate(metrics):
            topic = np.argmax(Whats[i], axis=1)
            coords_df['grp'] = topic
            print(f"Error is {metric[0]}.")
            print(f"Accuracy is {metric[1]}.")
            print(f"Time is {times[i]}")
            sns.scatterplot(x='x', y='y', hue='grp', data=coords_df, palette='viridis', ax=axes[i+1], legend=False)
            if i == 0:  
                axes[i+1].set_title(names[i+1], fontsize=24, fontweight='bold') 
            else:
                axes[i+1].set_title(names[i+1], fontsize=20)
        plt.tight_layout()
        plt.show()

    return W, Whats

def run_simul_fixed_lambda(nsim, lambd_grid, N=10, n=1000, p=30, K=3, r=0.05, m=5, phi=0.1):
    results = []

    for lambd in lambd_grid:
        print(f"Running simulation for lambda={lambd}...")

        for _ in range(nsim):
            # Generate topic data and graph
            while True:
                try:
                    coords_df, W, A, X = gen_model.generate_data(N, n, p, K, r)
                    weights, edge_df = gen_model.generate_weights_edge(coords_df, m, phi)
                    break
                except Exception as e:
                    print(f"An error has occurred: {e}")

            # Vanilla SVD
            start_time = time.time()
            model_v = splsi.SpLSI(lambd=lambd, lamb_start=0.0001, step_size=1.35, grid_len=30, method="nonspatial", verbose=0)
            model_v.fit(X, K, edge_df, weights)
            time_v = time.time() - start_time

            # Spatial SVD (two-step)
            start_time = time.time()
            model_splsi = splsi.SpLSI(lambd=lambd, lamb_start=0.0001, step_size=1.35, grid_len=30, step="two-step", verbose=0)
            model_splsi.fit(X, K, edge_df, weights)
            time_splsi = time.time() - start_time
            print(f"CV Lambda is {model_splsi.lambd}")

            # SLDA
            start_time = time.time()
            model_slda = utils.spatial_lda.model.run_simulation(X, K, coords_df)
            time_slda = time.time() - start_time
            
            # Record [err, acc] for spl_v and spl_2
            P_v = get_component_mapping(model_v.W_hat.T, W)
            P_splsi = get_component_mapping(model_splsi.W_hat.T, W)
            P_slda = get_component_mapping(model_slda.topic_weights.values.T, W)

            W_hat_v = model_v.W_hat @ P_v
            W_hat_plsi = model_splsi.W_hat @ P_splsi
            W_hat_slda = model_slda.topic_weights.values @ P_slda
            err_acc_spl_v = [get_F_err(W_hat_v, W), get_accuracy(coords_df, n, W_hat_v)]
            err_acc_spl_splsi = [get_F_err(W_hat_plsi, W), get_accuracy(coords_df, n, W_hat_plsi)]
            err_acc_spl_slda = [get_F_err(W_hat_slda, W), get_accuracy(coords_df, n, W_hat_slda)]

            results.append({
                'N': N,
                'vanilla_err': err_acc_spl_v[0],
                'vanilla_acc': err_acc_spl_v[1],
                'splsi_err': err_acc_spl_splsi[0],
                'splsi_acc': err_acc_spl_splsi[1],
                'slda_err': err_acc_spl_slda[0],
                'slda_acc': err_acc_spl_slda[1],
                'time_v': time_v,
                'time_splsi': time_splsi,
                'time_slda': time_slda,
                'model_v': model_v,
                'model_splsi': model_splsi,
                'model_slda': model_slda,
                'spatial_lambd': model_splsi.lambd
            })

    return results