from numpy.linalg import svd, norm
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
import numpy as np
import networkx as nx

def generate_graph_from_weights_(df, weights, n):
    np.random.seed(127)
    G = nx.Graph()
    
    for node in range(n):
        x = df['x'].iloc[node]
        y = df['y'].iloc[node]
        G.add_node(node, pos=(x, y))

    rows, cols = weights.nonzero()

    for i, j in zip(rows, cols):
        if i < j:
            w = weights[i, j]
            if w > 0:
                G.add_edge(i, j, weight=w)
    return G

def generate_graph_from_weights(df, weights, n):
    G = nx.Graph()
    for node in range(n):
        x = df['x'].iloc[node]
        y = df['y'].iloc[node]
        G.add_node(node, pos=(x, y))
    
    for node1 in G.nodes:
        for node2 in G.nodes:
            if node1 < node2:
                #pos1 = G.nodes[node1]['pos']
                #pos2 = G.nodes[node2]['pos']
                w = weights[node1,node2]
                #dist = norm(np.array(pos1) - np.array(pos2))
                if w > 0:
                    G.add_edge(node1, node2, weight=w)
    return G

def get_mst_path(G):
    mst = nx.minimum_spanning_tree(G)
    path = dict(nx.all_pairs_shortest_path_length(mst))
    return mst, path

def generate_mst(df, weights, n):
    G = generate_graph_from_weights(df, weights, n)
    mst, path = get_mst_path(G)
    return G, mst, path

def get_parent_node(mst, path, srn, nodenum):
    neighs = list(mst[nodenum].keys())
    length_to_srn = [path[neigh][srn] for neigh in neighs]
    parent = neighs[np.argmin(length_to_srn)]
    return parent

def interpolate_X(X, folds, foldnum, path, mst, srn):
    fold = folds[foldnum]
    
    X_tilde = X.copy()
    for node in fold:
        parent = get_parent_node(mst, path, srn, node)
        X_tilde[node,:] = X[parent,:]
    return X_tilde

def get_folds(mst, path, n, plot_tree=False):
    srn = np.random.choice(range(n),1)[0]
    print(f"Source node is {srn}")

    fold1 = []
    fold2 = []
    colors = []
    for key, value in path[srn].items():
        if (value%2)==0:
            fold1.append(key)
            colors.append("orange")
        elif (value%2)==1:
            fold2.append(key)
            colors.append("blue")
        else:
            colors.append("red")
    if plot_tree:
        nx.draw_kamada_kawai(mst, node_color = colors, node_size=10)
    return srn, fold1, fold2

def trunc_svd(X, K):
    U, L, VT = svd(X, full_matrices=False)
    U_k = U[:, :K]
    L_k = np.diag(L[:K])
    VT_k = VT[:K, :]
    return U_k, L_k, VT_k.T

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

def get_accuracy(df, n, W_hat):
    assgn = np.argmax(W_hat, axis=1)
    accuracy = np.sum(assgn == df['grp'].values) / n
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
