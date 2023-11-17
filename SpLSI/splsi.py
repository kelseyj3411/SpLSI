import numpy as np
from numpy.linalg import norm, svd, solve, qr
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment
import networkx as nx
import cvxpy as cp
from cvxpy import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.problems.problem import Problem
from SpLSI import generate_topic_model as gen_model
# use pycvxcluster "https://github.com/dx-li/pycvxcluster/tree/main"

def trunc_svd(X, K):
    U, L, VT = svd(X, full_matrices=False)
    U_k = U[:, :K]
    L_k = np.diag(L[:K])
    VT_k = VT[:K, :]
    return U_k, L_k, VT_k.T
    
def interpolate_X(X, folds, foldnum, path, mst, srn):
    fold = folds[foldnum]
    
    for node in fold:
        parent = gen_model.get_parent_node(path, mst, srn, node)
        X[node,:] = X[parent,:]
    return X

def update_U_tilde(X, L, V, weights, folds, path, mst, srn, lambd_grid, n, K):
    U_best_comb = np.zeros((n,K))
    lambds_best = {}

    for j in folds.keys():
        fold = folds[j]
        X_tilde = interpolate_X(X, folds, j, path, mst, srn)
        X_j = X_tilde[fold,:]

        best_err = float("inf")
        U_best = None
        lambd_best = 0
        
        for lambd in lambd_grid:
            XV = np.dot(X_tilde, V)
            ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd)
            ssnal.fit(X=XV, weight_matrix=weights, save_centers=True)
            U_hat = ssnal.centers_.T
            M = np.dot(np.dot(U_hat, L), V.T)
            err = norm(X_j-M[fold,:])
            if err < best_err:
                lambd_best = lambd
                U_best = U_hat

        U_best_comb[fold,:] = U_best[fold,:]
        lambds_best[j] = lambd_best

    best_err = float("inf")
    lambd_cv = 0

    for lambd in lambd_grid:
        XV = np.dot(X, V)
        ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd)
        ssnal.fit(X=XV, weight_matrix=weights, save_centers=True)
        U_hat_full = ssnal.centers_.T
        err = norm(U_hat_full-U_best_comb)
        if err < best_err:
            lambd_cv = lambd
            U_cv = U_hat_full
    Q, R = qr(U_cv)
    print(f"Best lambda is {lambd_cv}")
    return Q, lambd_cv

def update_V_tilde(X, U_tilde):
    V_hat = np.dot(X.T, U_tilde)
    Q, R = qr(V_hat)
    return Q

def update_L_tilde(X, U_tilde, V_tilde):
    L_tilde = np.dot(np.dot(U_tilde.T, X), V_tilde)
    return L_tilde

def run_iter_cvx(X, L, V, weights, folds, path, mst, srn, lambd_grid, n, K, eps):
    U, L, V = trunc_svd(X, K)
    thres = 1
    while thres > eps:
        UUT_old = np.dot(U, U.T)
        VVT_old = np.dot(V, V.T)

        U, lambd = update_U_tilde(X, L, V, weights, folds, path, mst, srn, lambd_grid, n, K)
        V = update_V_tilde(X, U)
        L = update_L_tilde(X, U, V)

        UUT = np.dot(U, U.T)
        VVT = np.dot(V, V.T)
        thres = np.max([norm(UUT-UUT_old)**2, norm(VVT-VVT_old)**2])
        print(thres)
    return U, lambd


def preprocess_U(U, K):
    for k in range(K):
        if U[0, k] < 0:
            U[:, k] = -1 * U[:, k]
    return U

def proj_simplex(v):
    n = len(v)
    if np.sum(v) == 1 and np.all(v >= 0):
        return v
    u = np.sort(v)[::-1]
    rho = np.max(np.where(u * np.arange(1, n + 1) > (np.cumsum(u) - 1)))
    theta = (np.cumsum(u) - 1) / rho
    w = np.maximum(v - theta, 0)
    return w

def get_component_mapping(stats_1, stats_2):
    similarity = np.dot(stats_1, stats_2.T)
    cost_matrix = -similarity
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    P = np.zeros_like(cost_matrix)
    P[row_ind, col_ind] = 1
    return P

def get_W_hat(U, H, n, K):
    Theta = Variable((n,K))
    constraints = [
        cp.sum(Theta[i, :]) == 1 for i in range(n)
    ]
    constraints += [
        Theta[i, j] >= 0 for i in range(n)
        for j in range(K)
    ]
    obj = Minimize(cp.norm(U - Theta @ H, 'fro'))
    prob = Problem(obj, constraints)
    prob.solve()
    return np.array(Theta.value)

def fit_SPOC(df, D, W, U, K, w, method="spatial"):
    if method != "spatial":
        print("Running vanilla SPOC...")
        X = D.T
        U, L, V = trunc_svd(X, K)

    J = []
    S = preprocess_U(U, K).T
    
    for t in range(K):
        maxind = np.argmax(norm(S, axis=0))
        s = np.reshape(S[:, maxind], (K, 1))
        S1 = (np.eye(K) - np.dot(s, s.T) / norm(s)**2).dot(S)
        S = S1
        J.append(maxind)
    
    H_hat = U[J, :]
    W_hat = get_W_hat(U, H_hat)

    P = get_component_mapping(W_hat.T, W)
    W_hat = np.dot(W_hat, P)
    
    assgn = np.argmax(W_hat, axis=1)
    accuracy = np.sum(assgn == df['grp'].values) / n
    err = norm(W.T - W_hat, ord='fro')
    print(err)
    return {'acc': accuracy, 'f.err': err, 'What': W_hat}