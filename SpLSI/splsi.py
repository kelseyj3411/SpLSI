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
# use pycvxcluster from "https://github.com/dx-li/pycvxcluster/tree/main"


class SpLSI(object):

    def __init__(
            self,
            lamb_start = 0.01,
            step_size = 1.15,
            grid_len = 100,
            eps = 1e-06,
            method = "spatial",
            return_anchor_docs = True,
            verbose = 1
    ):
        """
        Parameters
        -----------

        """
        self.lamb_start = lamb_start
        self.step_size = step_size
        self.grid_len = grid_len
        self.eps = eps
        self.method = method
        self.return_anchor_docs = return_anchor_docs
        self.verbose = verbose
        
    def fit(self, 
            D, 
            K, 
            df, 
            weights
    ):
        if self.method != "spatial":
            self.U, _, _ = trunc_svd(D.T, K)
            print("Running vanilla SVD...")
        
        else:
            print("Running spatial SVD...")
            self.U, self.lamd = spatialSVD(D, 
                                K, 
                                df, 
                                weights,
                                self.lamb_start,
                                self.step_size,
                                self.grid_len,
                                self.eps,
                                self.verbose
        )
        print("Running SPOC...")
        n = D.shape[1]
        J = []
        S = self.preprocess_U(self.U, K).T
        for t in range(K):
            maxind = np.argmax(norm(S, axis=0))
            s = np.reshape(S[:, maxind], (K, 1))
            S1 = (np.eye(K) - np.dot(s, s.T) / norm(s)**2).dot(S)
            S = S1
            J.append(maxind) 

        H_hat = self.U[J, :]
        self.W_hat = self.get_W_hat(self.U, H_hat, n, K)

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

        
    
