import sys
import random
import numpy as np
from numpy.linalg import norm, svd, solve, qr
import networkx as nx

from scipy.sparse.linalg import svds

from SpLSI.utils import *
from SpLSI import cfg

import pycvxcluster.pycvxcluster

# use pycvxcluster from "https://github.com/dx-li/pycvxcluster/tree/main"
from multiprocessing import Pool


def spatialSVD(
    X,
    K,
    edge_df,
    weights,
    lambd_fixed,
    lamb_start,
    step_size,
    grid_len,
    maxiter,
    eps,
    verbose,
    use_mpi,
    initialize
):
    n = X.shape[0]
    p = X.shape[1]
    srn, fold1, fold2, G, mst = get_folds_disconnected_G(edge_df)
    folds = {0: fold1, 1: fold2}

    lambd_grid = (lamb_start * np.power(step_size, np.arange(grid_len))).tolist()
    lambd_grid.insert(0, 1e-06)

    if initialize:
        M, _, _ = initial_svd(X, G, weights, folds, lambd_grid)
        U, L, V = svds(M, k=K)
        V  = V.T
        L = np.diag(L)
    else:
        U, L, V = svds(X, k=K)
        V  = V.T
        L = np.diag(L)

    score = 1
    niter = 0
    while score > eps and niter < maxiter:
        # Estimate the convergence metric
        if n > 1000:
            idx = np.random.choice(range(n),1000,replace=False)
        else:
            idx = range(n)
        
        U_samp = U[idx,:]
        UUT_old = np.dot(U_samp, U_samp.T)
        VVT_old = np.dot(V, V.T)

        if lambd_fixed is None:
            if use_mpi:
                 U, lambd, lambd_errs = update_U_tilde_mpi(X, V, G, weights, folds, lambd_grid, n, K)
            else:
                 U, lambd, lambd_errs = update_U_tilde(X, V, L, G, weights, folds, lambd_grid, n, K)
            V, L = update_V_L_tilde(X, U)
        else:
            U, lambd, lambd_errs = update_U_tilde_nocv(X, V, weights, lambd_fixed)
            V, L = update_V_L_tilde(X, U)

        UUT = np.dot(U[idx,:], U[idx,:].T)
        VVT = np.dot(V, V.T)
        score = np.max([(norm(UUT - UUT_old) ** 2)/UUT.shape[0], (norm(VVT - VVT_old) ** 2)/p])
        niter += 1
        if verbose == 1:
            print(f"Error is {score}")
        

    print(f"SpatialSVD ran for {niter} steps.")

    return U, V, L, lambd, lambd_errs, niter


def lambda_search_init(j, folds, X, G, weights, lambd_grid):
    fold = folds[j]
    X_tilde = interpolate_X(X, G, folds, j)
    X_j = X[fold, :]
  
    errs = []
    best_err = float("inf")
    M_best = None
    lambd_best = 0

    ssnal = pycvxcluster.pycvxcluster.SSNAL(verbose=0)

    for fitn, lambd in enumerate(lambd_grid):
        ssnal.gamma = lambd
        ssnal.fit(
            X=X_tilde,
            weight_matrix=weights,
            save_centers=True,
            save_labels=False,
            recalculate_weights=(fitn == 0),
        )
        ssnal.kwargs["x0"] = ssnal.centers_
        ssnal.kwargs["y0"] = ssnal.y_
        ssnal.kwargs["z0"] = ssnal.z_
        M_hat = ssnal.centers_.T
        err = norm(X_j - M_hat[fold, :])
        errs.append(err)
        if err < best_err:
            lambd_best = lambd
            M_best = M_hat
            best_err = err
    return j, errs, M_best, lambd_best


def initial_svd(X, G, weights, folds, lambd_grid):
    lambds_best = []
    lambd_errs = {"fold_errors": {}, "final_errors": []}
    
    with Pool(2) as p:
        results = p.starmap(
            lambda_search_init,
            [(j, folds, X, G, weights, lambd_grid) for j in folds.keys()],
        )
    for result in results:
        j, errs, _, lambd_best = result
        lambd_errs["fold_errors"][j] = errs
        lambds_best.append(lambd_best)

    cv_errs = np.add(lambd_errs["fold_errors"][0], lambd_errs["fold_errors"][1])
    lambd_cv = lambd_grid[np.argmin(cv_errs)]

    ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd_cv, verbose=0)
    ssnal.fit(X=X, weight_matrix=weights, save_centers=True)
    M_hat = ssnal.centers_.T

    print(f"Optimal lambda is {lambd_cv}...")
    return M_hat, lambd_cv, lambd_errs


def lambda_search_(j, folds, X, V, L, G, weights, lambd_grid):
    fold = folds[j]
    X_tilde = interpolate_X(X, G, folds, j)
    # print((X_tilde[fold[j],:]==X[fold[j],:]).sum()) # shouldn't be large
    # assert((X_tilde[fold[j],:]==X[fold[j],:]).sum()<=1)
    XV_tilde = (X_tilde @ V) @ np.diag(1/np.diag(L))
    X_j = X[fold, :]

    errs = []
    best_err = float("inf")
    U_best = None
    lambd_best = 0

    ssnal = pycvxcluster.pycvxcluster.SSNAL(verbose=0)

    for fitn, lambd in enumerate(lambd_grid):
        ssnal.gamma = lambd
        ssnal.fit(
            X=XV_tilde,
            weight_matrix=weights,
            save_centers=True,
            save_labels=False,
            recalculate_weights=(fitn == 0),
        )
        ssnal.kwargs["x0"] = ssnal.centers_
        ssnal.kwargs["y0"] = ssnal.y_
        ssnal.kwargs["z0"] = ssnal.z_
        U_hat = ssnal.centers_.T
        E = (U_hat @ L) @ V.T
        err = norm(X_j - E[fold, :])
        errs.append(err)
        if err < best_err:
            lambd_best = lambd
            U_best = U_hat
            best_err = err
    return j, errs, U_best, lambd_best


def lambda_search(j, folds, X, V, L, G, weights, lambd_grid):
    fold = folds[j]
    X_tilde = interpolate_X(X, G, folds, j)
    # print((X_tilde[fold[j],:]==X[fold[j],:]).sum()) # shouldn't be large
    # assert((X_tilde[fold[j],:]==X[fold[j],:]).sum()<=1)
    XV_tilde = np.dot(X_tilde, V)
    X_j = X[fold, :]

    errs = []
    best_err = float("inf")
    UL_best = None
    lambd_best = 0

    ssnal = pycvxcluster.pycvxcluster.SSNAL(verbose=0)

    for fitn, lambd in enumerate(lambd_grid):
        ssnal.gamma = lambd
        ssnal.fit(
            X=XV_tilde,
            weight_matrix=weights,
            save_centers=True,
            save_labels=False,
            recalculate_weights=(fitn == 0),
        )
        ssnal.kwargs["x0"] = ssnal.centers_
        ssnal.kwargs["y0"] = ssnal.y_
        ssnal.kwargs["z0"] = ssnal.z_
        UL_hat = ssnal.centers_.T
        E = np.dot(UL_hat, V.T)
        err = norm(X_j - E[fold, :])
        errs.append(err)
        if err < best_err:
            lambd_best = lambd
            UL_best = UL_hat
            best_err = err
    return j, errs, UL_best, lambd_best

def update_U_tilde_nocv(X, V, weights, lambd):
    XV = np.dot(X, V)
    ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd, verbose=0)
    ssnal.fit(X=XV, weight_matrix=weights, save_centers=True)

    U_hat = ssnal.centers_.T
    E = np.dot(U_hat, V.T)
    err = norm(X - E)

    Q, R = qr(U_hat)
    return Q, lambd, err

    
def update_U_tilde_mpi(X, V, G, weights, folds, lambd_grid, n, K):

    UL_best_comb = np.zeros((n, K))
    lambd_errs = {"fold_errors": {}, "final_errors": []}
    XV = np.dot(X, V)

    tasks = None
    if cfg.rank == 0:
        tasks = [(j, folds, X, V, G, weights, lambd_grid) for j in folds.keys()]

    if cfg.rank == 0 and len(tasks) < cfg.nproc:
        print("Number of tasks smaller than size.")
        tasks += [None] * (cfg.nproc - len(tasks))

    task = cfg.comm.scatter(tasks, root=0)

    if task is not None:
        j, errs, UL_best, lambd_best = lambda_search(*task)
        result = (j, errs, UL_best, lambd_best)
    else:
        result = None

    print("Gathering results...")
    results = cfg.comm.gather(result, root=0)

    if cfg.rank == 0:
        results = [res for res in results if res is not None]
        for j, errs, UL_best, lambd_best in results:
            lambd_errs["fold_errors"][j] = errs
            UL_best_comb[folds[j], :] = UL_best[folds[j], :]

        cv_errs = np.add(lambd_errs["fold_errors"][0], lambd_errs["fold_errors"][1])
        lambd_cv = lambd_grid[np.argmin(cv_errs)]

        ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd_cv, verbose=0)
        ssnal.fit(X=XV, weight_matrix=weights, save_centers=True)
        UL_hat_full = ssnal.centers_.T

        Q, R = qr(UL_hat_full)
        print("This is main rank.")
        return Q, lambd_cv, lambd_errs
    else:
        print("This is not main rank.")
        return None, None, None


def update_U_tilde(X, V, L, G, weights, folds, lambd_grid, n, K):
    lambds_best = []
    lambd_errs = {"fold_errors": {}, "final_errors": []}
    #XV = (X @ V) @ np.diag(np.diag(L)) 
    XV = X @ V

    with Pool(2) as p:
        results = p.starmap(
            lambda_search,
            [(j, folds, X, V, L, G, weights, lambd_grid) for j in folds.keys()],
        )
    for j, errs, _, lambd_best in results:
        lambd_errs["fold_errors"][j] = errs
        lambds_best.append(lambd_best)

    cv_errs = np.add(lambd_errs["fold_errors"][0], lambd_errs["fold_errors"][1])
    lambd_cv = lambd_grid[np.argmin(cv_errs)]

    ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd_cv, verbose=0)
    ssnal.fit(X=XV, weight_matrix=weights, save_centers=True)
    U_hat_full = ssnal.centers_.T

    Q, R = qr(U_hat_full)
    print(f"Optimal lambda is {lambd_cv}...")
    return Q, lambd_cv, lambd_errs


def update_V_L_tilde(X, U_tilde):
    V_hat = np.dot(X.T, U_tilde)
    Q, R = qr(V_hat)
    #L = np.diag(np.diag((U_tilde.T @ X) @ V_hat))
    # L = np.diag(np.diag(R))
    return Q, R
