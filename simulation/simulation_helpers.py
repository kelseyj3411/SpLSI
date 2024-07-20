import sys
import os
import time
import pickle
import numpy as np
from numpy.linalg import norm, svd, solve, qr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx

from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment

import cvxpy as cp
from cvxpy import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.problems.problem import Problem

import pycvxcluster.pycvxcluster

from SpLSI import generate_topic_model as gen_model
from SpLSI.utils import *
from SpLSI import splsi_
import utils.spatial_lda.model

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import numpy2ri

from collections import defaultdict

def _euclidean_proj_simplex(v, s=1):
        n = v.shape[0]
        # check if we are already on the simplex
        if v.sum() == s and np.alltrue(v >= 0):
            return v
        
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
       
        theta = (cssv[rho] - s) / (rho + 1.0)
        w = (v - theta).clip(min=0)
        return w

def calculate_norm(matrix):
    return norm(matrix, axis=1)


def run_simul(
    nsim=1,
    N=100,
    n=1000,
    p=30,
    K=3,
    rt=0.05,
    m=5,
    phi=0.1,
    lamb_start=0.001,
    step_size=1.2,
    grid_len=29,
    eps=1e-05,
    start_seed=None,
):
    results = defaultdict(list)
    models = {'hooi':[], 'splsi': [], 'plsi': [], 'tscore': [], 'lda': [], 'slda': []}

    # Activate automatic conversion of numpy objects to rpy2 objects
    numpy2ri.activate()

    # Import the necessary R packages
    nnls = rpackages.importr('nnls')
    rARPACK = rpackages.importr('rARPACK')
    quadprog = rpackages.importr('quadprog')
    Matrix = rpackages.importr('Matrix')

    r = robjects.r
    r['source']('uitls/topicscore.R')


    # print(f"Running simulations for N={N}, n={n}, p={p}, K={K}, r={r}, m={m}, phi={phi}...")
    for trial in range(nsim):
        os.system(f"echo Running trial {trial}...")
        # Generate topic data and graph
        regens = 0
        if not (start_seed is None):
            np.random.seed(start_seed + trial)
        while True:
            try:
                coords_df, W, A, X = gen_model.generate_data(N, n, p, K, rt, sparse=1)
                weights, edge_df = gen_model.generate_weights_edge(coords_df, m, phi)
                D = N * X.T

                # TopicScore
                start_time = time.time()
                Mquantile = 0
                K0 = int(np.ceil(1.5*K))
                c = min(10*K, int(np.ceil(D.shape[0]*0.7)))
                D_r = robjects.r.matrix(D, nrow=D.shape[0], ncol=D.shape[1])  # Convert to R matrix
                norm_score = r['norm_score']
                A_hat_ts = norm_score(K, K0, c, D_r)
                M = np.mean(D, axis=1)
                M_trunk = np.minimum(M, np.quantile(M, Mquantile))
                S = np.diag(np.sqrt(1/M_trunk))
                H = S @ A_hat_ts 
                projector = np.linalg.inv(H.T @ H) @ H.T
                theta_W = (projector @ S) @ D
                W_hat_ts= np.array([_euclidean_proj_simplex(x) for x in theta_W.T])
                time_ts = time.time() - start_time
                A_hat_ts = A_hat_ts.T

                # Vanilla SVD
                start_time = time.time()
                model_plsi = splsi_.SpLSI_(
                    method="nonspatial",
                    verbose=0,
                )
                model_plsi.fit(X, K, edge_df, weights)
                time_plsi = time.time() - start_time

                # Spatial SVD
                start_time = time.time()
                model_splsi = splsi_.SpLSI_(
                    lamb_start=lamb_start,
                    step_size=step_size,
                    grid_len=grid_len,
                    verbose=0,
                    eps=eps
                )
                model_splsi.fit(X, K, edge_df, weights)
                time_splsi = time.time() - start_time
                print(f"CV Lambda is {model_splsi.lambd}")

                # HOOI
                start_time = time.time()
                model_hooi = splsi_.SpLSI_(
                    lamb_start=lamb_start,
                    step_size=step_size,
                    grid_len=grid_len,
                    method="hooi",
                    verbose=0,
                    eps=eps,
                    normalize=True,
                    L_inv_=True
                )
                model_hooi.fit(X, K, edge_df, weights)
                time_hooi= time.time() - start_time
                print(f"CV Lambda is {model_hooi.lambd}")

                
                # LDA 
                print("Running LDA...")
                start_time = time.time()
                model_lda = LatentDirichletAllocation(n_components=K, topic_word_prior=1.1, random_state=0)
                model_lda.fit(D.T)
                W_hat_lda = model_lda.transform(D.T)
                A_hat_lda = model_lda.components_
                time_lda = time.time() - start_time

                # SLDA
                start_time = time.time()
                model_slda = utils.spatial_lda.model.run_simulation(X, K, coords_df)
                A_hat_slda = model_slda.components_
                row_sums = A_hat_slda.sum(axis=1, keepdims=True)
                A_hat_slda = (A_hat_slda / row_sums).T
                time_slda = time.time() - start_time
                break
            except Exception as e:
                print(f"Regenerating dataset due to error: {e}")
                regens += 1

        M = X[edge_df['src']] - X[edge_df['tgt']]
        s = sum(calculate_norm(M) * np.sqrt(edge_df['weight']))

        # Record [err, acc] for spl_v and spl_2
        P_plsi = get_component_mapping(model_plsi.W_hat.T, W)
        P_splsi = get_component_mapping(model_splsi.W_hat.T, W)
        P_hooi = get_component_mapping(model_hooi.W_hat.T, W)
        P_ts = get_component_mapping(W_hat_ts.T, W)
        P_lda = get_component_mapping(W_hat_lda.T, W)
        P_slda = get_component_mapping(model_slda.topic_weights.values.T, W)

        W_hat_plsi = model_plsi.W_hat @ P_plsi
        W_hat_splsi = model_splsi.W_hat @ P_splsi
        W_hat_hooi = model_hooi.W_hat @ P_hooi
        W_hat_ts = W_hat_ts @ P_ts
        W_hat_lda = W_hat_lda @ P_lda
        W_hat_slda = model_slda.topic_weights.values @ P_slda

        A_hat_plsi = P_plsi.T @ model_plsi.A_hat
        A_hat_splsi = P_splsi.T @ model_splsi.A_hat
        A_hat_hooi = P_hooi.T @ model_hooi.A_hat
        A_hat_ts = P_ts.T @ A_hat_ts
        A_hat_lda = P_lda.T @ A_hat_lda
        A_hat_slda = P_slda.T @ A_hat_slda.T

        err_acc_spl_plsi = [
            get_F_err(W_hat_plsi, W),
            get_l1_err(W_hat_plsi, W),
            get_F_err(A_hat_plsi, A),
            get_l1_err(A_hat_plsi, A),
            get_accuracy(coords_df, n, W_hat_plsi),
        ]
        err_acc_spl_splsi = [
            get_F_err(W_hat_splsi, W),
            get_l1_err(W_hat_splsi, W),
            get_F_err(A_hat_splsi, A),
            get_l1_err(A_hat_splsi, A),
            get_accuracy(coords_df, n, W_hat_splsi),
        ]
        err_acc_spl_hooi = [
            get_F_err(W_hat_hooi, W),
            get_l1_err(W_hat_hooi, W),
            get_F_err(A_hat_hooi, A),
            get_l1_err(A_hat_hooi, A),
            get_accuracy(coords_df, n, W_hat_hooi),
        ]
        err_acc_spl_ts = [
            get_F_err(W_hat_ts, W),
            get_l1_err(W_hat_ts, W),
            get_F_err(A_hat_ts, A),
            get_l1_err(A_hat_ts, A),
            get_accuracy(coords_df, n, W_hat_ts),
        ]
        err_acc_spl_lda = [
            get_F_err(W_hat_lda, W),
            get_l1_err(W_hat_lda, W),
            get_F_err(A_hat_lda, A),
            get_l1_err(A_hat_lda, A),
            get_accuracy(coords_df, n, W_hat_lda),
        ]
        err_acc_spl_slda = [
            get_F_err(W_hat_slda, W),
            get_l1_err(W_hat_slda, W),
            get_F_err(A_hat_slda, A),
            get_l1_err(A_hat_slda, A),
            get_accuracy(coords_df, n, W_hat_slda),
        ]

        results["trial"].append(trial)
        results["jump"].append(s/n)
        results["N"].append(N)
        results["n"].append(n)
        results["p"].append(p)
        results["K"].append(K)

        results["plsi_err"].append(err_acc_spl_plsi[0])
        results["plsi_l1_err"].append(err_acc_spl_plsi[1])
        results["plsi_acc"].append(err_acc_spl_plsi[4])

        results["splsi_err"].append(err_acc_spl_splsi[0])
        results["splsi_l1_err"].append(err_acc_spl_splsi[1])
        results["splsi_acc"].append(err_acc_spl_splsi[4])

        results["hooi_err"].append(err_acc_spl_hooi[0])
        results["hooi_l1_err"].append(err_acc_spl_hooi[1])
        results["hooi_acc"].append(err_acc_spl_hooi[4])

        results["ts_err"].append(err_acc_spl_ts[0])
        results["ts_l1_err"].append(err_acc_spl_ts[1])
        results["ts_acc"].append(err_acc_spl_ts[4])

        results["lda_err"].append(err_acc_spl_lda[0])
        results["lda_l1_err"].append(err_acc_spl_lda[1])
        results["lda_acc"].append(err_acc_spl_lda[4])

        results["slda_err"].append(err_acc_spl_slda[0])
        results["slda_l1_err"].append(err_acc_spl_slda[1])
        results["slda_acc"].append(err_acc_spl_slda[4])

        results["plsi_time"].append(time_plsi)
        results["splsi_time"].append(time_splsi)
        results["hooi_time"].append(time_hooi)
        results["ts_time"].append(time_ts)
        results["lda_time"].append(time_lda)
        results["slda_time"].append(time_slda)
        results["spatial_lambd"].append(model_splsi.lambd)
        results["splsi_iters"].append(model_splsi.used_iters)

        results["A_plsi_err"].append(err_acc_spl_plsi[2])
        results["A_plsi_l1_err"].append(err_acc_spl_plsi[3])
        results["A_splsi_err"].append(err_acc_spl_splsi[2])
        results["A_splsi_l1_err"].append(err_acc_spl_splsi[3])
        results["A_hooi_err"].append(err_acc_spl_hooi[2])
        results["A_hooi_l1_err"].append(err_acc_spl_hooi[3])
        results["A_ts_err"].append(err_acc_spl_ts[2])
        results["A_ts_l1_err"].append(err_acc_spl_ts[3])
        results["A_lda_err"].append(err_acc_spl_lda[2])
        results["A_lda_l1_err"].append(err_acc_spl_lda[3])
        results["A_slda_err"].append(err_acc_spl_slda[2])
        results["A_slda_l1_err"].append(err_acc_spl_slda[3])

        models['plsi'].append(model_plsi)
        models['splsi'].append(model_splsi)
        models['hooi'].append(model_hooi)
        models['tscore'].append(A_hat_ts.T)
        models['lda'].append(model_lda)
        models['slda'].append(model_slda)

        model_save_loc = os.path.join(os.getcwd(), "sim_models")
        file_base = os.path.join(model_save_loc, f"simul_N={N}_n={n}_K={K}_p={p}")
        extensions = ['.csv', '.pkl']
        if not os.path.exists(model_save_loc):
            os.makedirs(model_save_loc)
        pkl_loc = f"{file_base}_trial={trial}{extensions[1]}"
        with open(pkl_loc, "wb") as f:
            pickle.dump(models, f)

    results = pd.DataFrame(results)
    return results
