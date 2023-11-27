import sys
import numpy as np
from numpy.linalg import norm, svd, solve, qr
import networkx as nx

from SpLSI.utils import *
sys.path.append('./SpLSI/pycvxcluster/src/')
import pycvxcluster.pycvxclt
# use pycvxcluster from "https://github.com/dx-li/pycvxcluster/tree/main"


def spatialSVD(
        D,
        K,
        df,
        weights,
        lamb_start,
        step_size,
        grid_len,
        eps,
        verbose
):
    X = D.T
    n = X.shape[0]
    _, mst, path = generate_mst(df, weights, n)
    srn, fold1, fold2 = get_folds(mst, path, n)
    folds = {0:fold1, 1:fold2}

    lambd_grid = (lamb_start*np.power(step_size, np.arange(grid_len))).tolist()

    U, _, V = trunc_svd(X, K)
    thres = 1
    while thres > eps:
        UUT_old = np.dot(U, U.T)
        VVT_old = np.dot(V, V.T)

        U, lambd = update_U_tilde(X, V, weights, folds, path, mst, srn, lambd_grid, n, K)
        V = update_V_tilde(X, U)
        #L = update_L_tilde(X, U, V)

        UUT = np.dot(U, U.T)
        VVT = np.dot(V, V.T)
        thres = np.max([norm(UUT-UUT_old)**2, norm(VVT-VVT_old)**2])
        if verbose == 1:
            print(f"Error is {thres}")
    return U, lambd


def update_U_tilde(X, V, weights, folds, path, mst, srn, lambd_grid, n, K):
    U_best_comb = np.zeros((n,K))
    lambds_best = {}

    for j in folds.keys():
        fold = folds[j]
        X_tilde = interpolate_X(X, folds, j, path, mst, srn)
        # print((X_tilde[fold[j],:]==X[fold[j],:]).sum()) # shouldn't be large
        #assert((X_tilde[fold[j],:]==X[fold[j],:]).sum()<=1)
        X_j = X[fold,:]
        XV = np.dot(X_tilde, V)

        best_err = float("inf")
        U_best = None
        lambd_best = 0
        
        for lambd in lambd_grid:
            ssnal = pycvxcluster.pycvxclt.SSNAL(gamma=lambd, verbose=0)
            ssnal.fit(X=XV, weight_matrix=weights, save_centers=True)
            U_hat = ssnal.centers_.T
            row_sums = norm(U_hat, axis=1, keepdims=True)
            U_hat = U_hat / row_sums
            M = np.dot(U_hat, V.T)
            err = norm(X_j-M[fold,:])
            if err < best_err:
                lambd_best = lambd
                U_best = U_hat
                best_err = err
        U_best_comb[fold,:] = U_best[fold,:]
        lambds_best[j] = lambd_best

    errs = []
    best_err = float("inf")
    lambd_cv = 0
    XV = np.dot(X, V)
    for lambd in lambd_grid:
        ssnal = pycvxcluster.pycvxclt.SSNAL(gamma=lambd, verbose=0)
        ssnal.fit(X=XV, weight_matrix=weights, save_centers=True)
        U_hat_full = ssnal.centers_.T
        row_sums = norm(U_hat_full, axis=1, keepdims=True)
        U_hat_full = U_hat_full / row_sums
        err = norm(U_hat_full-U_best_comb)
        errs.append(err)
        if err < best_err:
            lambd_cv = lambd
            U_cv = U_hat_full
            best_err = err
    Q, R = qr(U_cv)
    return Q, lambd_cv


def update_V_tilde(X, U_tilde):
    V_hat = np.dot(X.T, U_tilde)
    row_sums = norm(V_hat, axis=1, keepdims=True)
    V_hat = V_hat / row_sums
    Q, R = qr(V_hat)
    return Q

def update_L_tilde(X, U_tilde, V_tilde):
    L_tilde = np.dot(np.dot(U_tilde.T, X), V_tilde)
    return L_tilde