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
from SpLSI.utils import *
from SpLSI.spatialSVD import *


class SpLSI(object):
    def __init__(
        self,
        lambd=None,
        lamb_start=0.01,
        step_size=1.15,
        grid_len=100,
        maxiter=25,
        eps=1e-08,
        method="spatial",
        step="two-step",
        return_anchor_docs=True,
        verbose=1,
    ):
        """
        Parameters
        -----------

        """
        self.lambd = lambd
        self.lamb_start = lamb_start
        self.step_size = step_size
        self.grid_len = grid_len
        self.maxiter = maxiter
        self.eps = eps
        self.method = method
        self.return_anchor_docs = return_anchor_docs
        self.verbose = verbose
        self.step = step

    def fit(self, X, K, edge_df, weights):
        if self.method != "spatial":
            self.U, self.L, self.V = trunc_svd(X, K)
            print("Running vanilla SVD...")

        else:
            print("Running spatial SVD...")
            (
                self.U,
                self.V,
                self.L,
                self.lambd,
                self.lambd_errs,
                self.used_iters,
            ) = spatialSVD(
                X,
                K,
                edge_df,
                weights,
                self.lambd,
                self.lamb_start,
                self.step_size,
                self.grid_len,
                self.maxiter,
                self.eps,
                self.verbose,
                self.step,
            )
        print("Running SPOC...")
        n = X.shape[0]
        J = []
        S = self.preprocess_U(self.U, K).T
        for t in range(K):
            maxind = np.argmax(norm(S, axis=0))
            s = np.reshape(S[:, maxind], (K, 1))
            S1 = (np.eye(K) - np.dot(s, s.T) / norm(s) ** 2).dot(S)
            S = S1
            J.append(maxind)

        H_hat = self.U[J, :]
        self.W_hat = self.get_W_hat(self.U, H_hat)
        M = self.U @ self.V.T
        self.A_hat = self.get_A_hat(self.W_hat, M)

        if self.return_anchor_docs:
            self.anchor_indices = J

        return self

    @staticmethod
    def preprocess_U(U, K):
        for k in range(K):
            if U[0, k] < 0:
                U[:, k] = -1 * U[:, k]
        return U

    @staticmethod
    def get_W_hat_cvx(U, H, n, K):
        Theta = Variable((n, K))
        constraints = [cp.sum(Theta[i, :]) == 1 for i in range(n)]
        constraints += [Theta[i, j] >= 0 for i in range(n) for j in range(K)]
        obj = Minimize(cp.norm(U - Theta @ H, "fro"))
        prob = Problem(obj, constraints)
        prob.solve()
        return np.array(Theta.value)

    def get_W_hat(self, U, H):
        projector = H.T.dot(np.linalg.inv(H.dot(H.T)))
        theta = U.dot(projector)
        theta_simplex_proj = np.array([self._euclidean_proj_simplex(x) for x in theta])
        return theta_simplex_proj

    def get_A_hat(self, W_hat, M):
        projector = (np.linalg.inv(W_hat.T.dot(W_hat))).dot(W_hat.T)
        theta = projector.dot(M)
        theta_simplex_proj = np.array([self._euclidean_proj_simplex(x) for x in theta])
        return theta_simplex_proj

    @staticmethod
    def _euclidean_proj_simplex(v, s=1):
        (n,) = v.shape
        # check if we are already on the simplex
        if v.sum() == s and np.alltrue(v >= 0):
            # best projection: itself!
            return v
        # get the array of cumulative sums of a sorted (decreasing) copy of v
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        # get the number of > 0 components of the optimal solution
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
        # compute the Lagrange multiplier associated to the simplex constraint
        theta = (cssv[rho] - s) / (rho + 1.0)
        # compute the projection by thresholding v using theta
        w = (v - theta).clip(min=0)
        return w
