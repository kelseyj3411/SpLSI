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


def lambda_search(j, folds, X, V, G, weights, lambd_grid):
    fold = folds[j]
    X_tilde = interpolate_X(X, G, folds, j)
    # print((X_tilde[fold[j],:]==X[fold[j],:]).sum()) # shouldn't be large
    # assert((X_tilde[fold[j],:]==X[fold[j],:]).sum()<=1)
    XV_tilde = np.dot(X_tilde, V)
    #L_hat = np.diag(norm(XV_tilde, axis=0))
    #XV_tilde = np.apply_along_axis(lambda x: x/norm(x), 0, XV_tilde)
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
        UL_hat = ssnal.centers_.T
        #U_hat, _ = qr(UL_hat_lambd)
        E = UL_hat @ V.T
        err = norm(X_j - E[fold, :])
        errs.append(err)
        if err < best_err:
            lambd_best = lambd
            U_best = UL_hat
            best_err = err
    return j, errs, U_best, lambd_best


def update_U_tilde(X, V, G, weights, folds, lambd_grid, n, K):
    lambds_best = []
    lambd_errs = {"fold_errors": {}, "final_errors": []}
    XV = np.dot(X, V)
    #XV = np.apply_along_axis(lambda x: x/norm(x), 0, XV)

    with Pool(2) as p:
        results = p.starmap(
            lambda_search,
            [(j, folds, X, V, G, weights, lambd_grid) for j in folds.keys()],
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
    #L = np.diag(np.diag(R))
    print(f"Optimal lambda is {lambd_cv}...")
    return Q, R, lambd_cv, lambd_errs


def gamma_search(j, folds, X, U, L, gamm_grid):
    fold = folds[j]
    fold_c = [item for i, fold in enumerate(folds) if i != j for item in fold]
    XTU_tilde = (X[fold_c,:].T @ U[fold_c,:]) @ np.diag(1/np.diag(L))
    #XTU_tilde = X[fold_c,:].T @ U[fold_c,:]
    XTU_tilde_row_norms = norm(XTU_tilde, axis=1)
    #L_hat = np.diag(norm(XTU_tilde, axis=0))
    #XTU_tilde = np.apply_along_axis(lambda x: x/norm(x), 0, XTU_tilde)
    X_j = X[fold, :]

    errs = []
    best_err = float("inf")
    gamm_best = 0

    for gamma in gamm_grid:
        #V_hat = np.apply_along_axis(soft_threshold, 0, XTU_tilde, gamma)
        #E = (U[fold,:] @ L) @ V_hat.T 
        #err = norm(X_j - E)
        #err = norm(V_hat @ L - X_j.T @ U[fold,:])
        #V_hat, _ = qr(V_hat_gamma)
        #err = norm(V_hat @ L_hat - X_j.T @ U[fold,:])
        thr = hard_threshold(XTU_tilde_row_norms, gamma)
        V_hat_thr = thr[:,None]*(np.apply_along_axis(lambda x: x/norm(x), 1, XTU_tilde))
        #V_hat,_ = qr(V_hat_thr)
        #E = (U @ L) @ V_hat.T
        err = norm(V_hat_thr @ L - X_j.T @ U[fold,:])
        print(err)
        #err = norm(X_j - E[fold, :])
        errs.append(err)
        if err < best_err:
            gamm_best = gamma
            best_err = err
    return j, errs, gamm_best


def update_V_tilde_(X, U_tilde, L, gamm_grid):
    gamms_best = []
    gamm_errs = []
    XTU = (X.T @ U_tilde) @ np.diag(1/np.diag(L))
    #XTU = X.T @ U_tilde
    XTU_row_norms = norm(XTU, axis=1)
    #XTU = np.apply_along_axis(lambda x: x/norm(x), 0, XTU)

    n = X.shape[0]
    folds = get_Kfolds(n, 3)

    with Pool(5) as p:
        results = p.starmap(
            gamma_search,
            [(j, folds, X, U_tilde, L, gamm_grid) for j in range(3)],
        )
    for j, errs, gamm_best in results:
        gamm_errs.append(errs)
        gamms_best.append(gamm_best)

    cv_errs = np.sum(gamm_errs, axis=0)
    gamm_cv = gamm_grid[np.argmin(cv_errs)]


    thr = hard_threshold(XTU_row_norms, gamm_cv)
    V_hat_full = thr[:,None]*(np.apply_along_axis(lambda x: x/norm(x), 1, XTU))

    Q, _ = qr(V_hat_full)
    print(f"Optimal gamma is {gamm_cv}...")
    return Q, gamm_errs, gamms_best, cv_errs, gamm_cv