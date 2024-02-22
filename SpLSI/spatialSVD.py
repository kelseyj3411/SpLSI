import sys
import numpy as np
from numpy.linalg import norm, svd, solve, qr
import networkx as nx

from SpLSI.utils import *

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
        method
):
    n = X.shape[0]
    p = X.shape[1]
    srn, fold1, fold2, G, mst = get_folds_disconnected_G(edge_df)
    folds = {0:fold1, 1:fold2}

    lambd_grid = (lamb_start*np.power(step_size, np.arange(grid_len))).tolist()
    lambd_grid.insert(0, 1e-06)

    if method == 'two-step':
        U, _, V = trunc_svd(X, K)

        thres = 1
        niter = 0
        while thres > eps*n and niter < maxiter:
            UUT_old = np.dot(U, U.T)
            VVT_old = np.dot(V, V.T)

            if lambd_fixed is None:
                U, lambd, lambd_errs = update_U_tilde(X, V, G, weights, folds, lambd_grid, n, K)
                V, L = update_V_L_tilde(X, U)
            else:
                U, lambd, lambd_errs = update_U_tilde_nocv(X, V, weights, lambd_fixed)
                V, L = update_V_L_tilde(X, U)

            UUT = np.dot(U, U.T)
            VVT = np.dot(V, V.T)
            thres = np.max([norm(UUT-UUT_old)**2, norm(VVT-VVT_old)**2])
            niter += 1
            if verbose == 1:
                print(f"Error is {thres}")
    else:
        M_tilde, lambd = update_M_tilde(X, G, weights, folds, lambd_grid, n, p)
        U, _, _ = trunc_svd(M_tilde, K)

    print(f"SpatialSVD ran for {niter} steps.")

    return U, V, L, lambd, lambd_errs


def update_M_tilde(X, G, weights, folds, lambd_grid, n, p):
    M_best_comb = np.zeros((n,p))
    lambds_best = {}
    lambd_errs = {'fold_errors': {}, 'final_errors': []}

    for j in folds.keys():
        fold = folds[j]
        X_tilde = interpolate_X(X, G, folds, j)
        # print((X_tilde[fold[j],:]==X[fold[j],:]).sum()) # shouldn't be large
        #assert((X_tilde[fold[j],:]==X[fold[j],:]).sum()<=1)
        X_j = X[fold,:]

        errs = []
        best_err = float("inf")
        M_best = None
        lambd_best = 0
        
        for lambd in lambd_grid:
            ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd, verbose=0)
            ssnal.fit(X=X_tilde, weight_matrix=weights, save_centers=True)
            M_hat = ssnal.centers_.T
            err = norm(X_j-M_hat[fold,:])
            if err < best_err:
                lambd_best = lambd
                M_best = M_hat
                best_err = err
        lambd_errs['fold_errors'][j] = errs
        M_best_comb[fold,:] = M_best[fold,:]
        lambds_best.append(lambd_best)

    cv_errs = np.add(lambd_errs['fold_errors'][0],lambd_errs['fold_errors'][1])
    lambd_cv = lambd_grid[np.argmin(cv_errs)]

    ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd_cv, verbose=0)
    ssnal.fit(X=X, weight_matrix=weights, save_centers=True)
    M_hat_full = ssnal.centers_.T

    Q, R = qr(M_hat_full)
    return Q, lambd_cv, lambd_errs


def update_U_tilde_nocv(X, V, weights, lambd):
    XV = np.dot(X, V)
    ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd, verbose=0)
    ssnal.fit(X=XV, weight_matrix=weights, save_centers=True)

    U_hat = ssnal.centers_.T
    E = np.dot(U_hat, V.T)
    err = norm(X-E)

    Q, R = qr(U_hat)
    return Q, lambd, err


def update_U_tilde(X, V, G, weights, folds, lambd_grid, n, K):
    UL_best_comb = np.zeros((n,K))
    lambds_best = []
    lambd_errs = {'fold_errors': {}, 'final_errors': []}
    XV = np.dot(X, V)

    with Pool(2) as p:
        results = p.starmap(lambda_search, [(j, folds, X, V, G, weights, lambd_grid) for j in folds.keys()])
    for j, errs, UL_best, lambd_best in results:

        lambd_errs['fold_errors'][j] = errs
        UL_best_comb[folds[j],:] = UL_best[folds[j],:]
        lambds_best.append(lambd_best)
    
    cv_errs = np.add(lambd_errs['fold_errors'][0],lambd_errs['fold_errors'][1])
    lambd_cv = lambd_grid[np.argmin(cv_errs)]

    ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd_cv, verbose=0)
    ssnal.fit(X=XV, weight_matrix=weights, save_centers=True)
    UL_hat_full = ssnal.centers_.T

    Q, R = qr(UL_hat_full)
    return Q, lambd_cv, lambd_errs

def lambda_search(j, folds, X, V, G, weights, lambd_grid):
    fold = folds[j]
    X_tilde = interpolate_X(X, G, folds, j)
    # print((X_tilde[fold[j],:]==X[fold[j],:]).sum()) # shouldn't be large
    #assert((X_tilde[fold[j],:]==X[fold[j],:]).sum()<=1)
    XV_tilde = np.dot(X_tilde, V)
    X_j = X[fold,:]

    errs = []
    best_err = float("inf")
    UL_best = None
    lambd_best = 0

    ssnal = pycvxcluster.pycvxcluster.SSNAL(verbose=0)
    
    for fitn, lambd in enumerate(lambd_grid):
        ssnal.gamma = lambd
        ssnal.fit(X=XV_tilde, weight_matrix=weights, save_centers=True, save_labels = False, recalculate_weights=(fitn == 0))
        ssnal.kwargs['x0'] = ssnal.centers_
        ssnal.kwargs['y0'] = ssnal.y_
        ssnal.kwargs['z0'] = ssnal.z_
        UL_hat = ssnal.centers_.T
        E = np.dot(UL_hat, V.T)
        err = norm(X_j-E[fold,:])
        errs.append(err)
        if err < best_err:
            lambd_best = lambd
            UL_best = UL_hat
            best_err = err
    return j, errs, UL_best, lambd_best

def update_U_tilde_old(X, V, G, weights, folds, path, mst, srn, lambd_grid, n, K):
    U_best_comb = np.zeros((n,K))
    lambds_best = []
    lambd_errs = {'fold_errors': {}, 'final_errors': []}
    XV = np.dot(X, V)

    for j in folds.keys():
        fold = folds[j]
        X_tilde = interpolate_X(X, G, folds, j, path, mst, srn)
        # print((X_tilde[fold[j],:]==X[fold[j],:]).sum()) # shouldn't be large
        #assert((X_tilde[fold[j],:]==X[fold[j],:]).sum()<=1)
        XV_tilde = np.dot(X_tilde, V)
        X_j = X[fold,:]

        errs = []
        best_err = float("inf")
        U_best = None
        lambd_best = 0
        
        for lambd in lambd_grid:
            ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd, verbose=0)
            ssnal.fit(X=XV_tilde, weight_matrix=weights, save_centers=True)
            U_hat = ssnal.centers_.T
            E = np.dot(U_hat, V.T)
            err = norm(X_j-E[fold,:])
            errs.append(err)
            if err < best_err:
                lambd_best = lambd
                U_best = U_hat
                best_err = err
        lambd_errs['fold_errors'][j] = errs
        U_best_comb[fold,:] = U_best[fold,:]
        lambds_best.append(lambd_best)

    final_errs = []
    best_final_err = float("inf")
    lambd_cv = 0
    for lambd in lambd_grid:
        ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd, verbose=0)
        ssnal.fit(X=XV, weight_matrix=weights, save_centers=True)
        U_hat_full = ssnal.centers_.T
        E_best = np.dot(U_best, V.T)
        E_full = np.dot(U_hat_full, V.T)
        final_err = norm(E_full-E_best)
        final_errs.append(final_err)
        if final_err < best_final_err:
            lambd_cv = lambd
            U_cv = U_hat_full
            best_final_err = final_err
    lambd_errs['final_errors'] = final_errs

    Q, R = qr(U_cv)
    return Q, lambd_cv, lambd_errs

def update_V_L_tilde(X, U_tilde):
    V_hat = np.dot(X.T, U_tilde)
    Q, R = qr(V_hat)
    return Q, R

