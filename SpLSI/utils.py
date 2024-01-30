from numpy.linalg import svd, norm
from scipy.optimize import linear_sum_assignment
import numpy as np
import networkx as nx

def get_mst(edge_df):
    G = nx.from_pandas_edgelist(edge_df, 'src', 'tgt')
    connected_subgraphs = list(nx.connected_components(G))
    mst = nx.minimum_spanning_tree(G)
    if len(connected_subgraphs) > 1:
        raise ValueError("Graph is not connected.")
    return G, mst

def get_shortest_paths(mst, srn):
    shortest_paths = dict(nx.shortest_path_length(mst, source=srn))
    return shortest_paths

def get_folds(mst):
    srn = np.random.choice(mst.nodes)
    path = get_shortest_paths(mst, srn)
    fold1 = [key for key, value in path.items() if value % 2 == 0]
    fold2 = [key for key, value in path.items() if value % 2 == 1]
    return srn, fold1, fold2

def interpolate_X(X, G, folds, foldnum):
    fold = folds[foldnum]
    
    X_tilde = X.copy()
    for node in fold:
        neighs = list(G.neighbors(node))
        neighs = list(set(neighs) - set(fold))
        X_tilde[node,:] = np.mean(X[neighs,:], axis=0)
    return X_tilde

def trunc_svd(X, K):
    U, L, VT = svd(X, full_matrices=False)
    U_k = U[:, :K]
    L_k = np.diag(L[:K])
    VT_k = VT[:K, :]
    return U_k, L_k, VT_k.T

def normaliza_coords(coords):
    """
    Input: pandas dataframe (n x 2) of x,y coordinates
    Output: pandas dataframe (n x 2) of normalizaed (0,1) x,y coordinates
    """
    minX = min(coords['x'])
    maxX = max(coords['x'])
    minY = min(coords['y'])
    maxY = max(coords['y'])
    diaglen = np.sqrt((minX-maxX)**2+(minY-maxY)**2)
    coords['x'] = (coords['x']-minX)/diaglen
    coords['y'] = (coords['y']-minY)/diaglen
    return coords.values

def dist_to_weight(df, coords, phi):
    """
    Input: 
    - df: pandas dataframe (n x 2) of src, dst nodes 
    - coords: pandas dataframe (n x 2) of normalizaed (0,1) x,y coordinates
    - phi: weight parameter
    Ouput: pandas dataframe (n x 3) of src, dst, squared exponential kernel distance
    """
    diff = coords.loc[df['src']].values-coords.loc[df['dst']].values
    w =  np.exp(-0.1 * np.apply_along_axis(norm, 1, diff)**2)
    return w

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

def get_accuracy(coords_df, n, W_hat):
    assgn = np.argmax(W_hat, axis=1)
    accuracy = np.sum(assgn == coords_df['grp'].values) / n
    return accuracy

def get_F_err(W, W_hat):
    err = norm(W.T - W_hat, ord='fro')
    return err

def inverse_L(L):
    d = np.diagonal(L)
    non_zero = d != 0
    inv_d = np.zeros_like(d)
    inv_d[non_zero] = 1.0/d[non_zero]
    inv = np.diag(inv_d)
    return L

