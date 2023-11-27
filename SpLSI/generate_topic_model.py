import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
import networkx as nx


def get_initial_centers(val, centers):
    quantiles = []
    for i in range(centers):
        quantiles.append(i * int(val.shape[0]/centers))
    return quantiles

def align_order(k, K):
    order = np.zeros(K, dtype=int)
    order[np.where(np.arange(K) != k)[0]] = np.random.choice(np.arange(1, K), K-1, replace=False)
    order[k] = 0
    return order

def reorder_with_noise(v, order, K, r):
    u = np.random.rand()
    if u < r:
        return v[order[np.random.choice(range(K), K, replace=False)]]
    else:
        sorted_row = np.sort(v)[::-1]
        return sorted_row[order]
    
def sample_MN(p, N):
    return np.random.multinomial(N, p, size=1)

def generate_graph(N, n, p, K, r):
    coords = np.zeros((n, 2))
    coords[:, 0] = np.random.uniform(0, 1, n)
    coords[:, 1] = np.random.uniform(0, 1, n)

    cluster_obj = KMeans(n_clusters=20, init=coords[get_initial_centers(coords, 20), :], n_init=1)
    grps = cluster_obj.fit_predict(coords)

    df = pd.DataFrame(coords, columns=['x','y'])
    df['grp'] = grps % K
    return df

def generate_W(df, N, n, p, K, r):
    W = np.zeros((K, n))
    for k in range(K):
        alpha = np.random.uniform(0.1, 0.5, K)
        cluster_size = df[df['grp'] == k].shape[0]
        order = align_order(k, K)
        inds = df['grp'] == k
        W[:, inds] = np.transpose(np.apply_along_axis(reorder_with_noise, 1, np.random.dirichlet(alpha, size=cluster_size), order, K, r))

        # generate pure doc 
        cano_ind = np.random.choice(np.where(inds)[0], 1)
        W[:, cano_ind] = np.eye(K)[0, :].reshape(K,1)
    return W

def generate_A(df, N, n, p, K, r):
    A = np.random.uniform(0, 1, size=(p, K))

    # generate pure word
    cano_ind = np.random.choice(np.arange(p), K, replace=False)
    A[cano_ind, :] = np.eye(K)
    A = np.apply_along_axis(lambda x: x / np.sum(x), 0, A)
    return A

def generate_data(N, n, p, K, r):
    df = generate_graph(N, n, p , K, r)
    W = generate_W(df, N, n, p , K, r)
    A = generate_A(df, N, n, p , K, r)
    D0 = np.dot(A, W)
    D = np.apply_along_axis(sample_MN, 0, D0, N).reshape(p,n)
    assert np.sum(np.apply_along_axis(np.sum, 0, D)!=N) == 0
    D = D/N

    return df, W, A, D

def generate_weights(df, K, nearest_n, phi):
    K = rbf_kernel(df[['x','y']], gamma = phi)
    np.fill_diagonal(K, 0)
    weights = np.zeros_like(K)

    for i in range(K.shape[0]):
        top_indices = np.argpartition(K[i], -nearest_n)[-nearest_n:]
        weights[i, top_indices] = K[i, top_indices]
        
    weights = (weights+weights.T)/2  
    # Adj = csr_matrix(weights)
    return weights

def plot_scatter(df):
    unique_groups = df['grp'].unique()
    cmap = plt.get_cmap('Set3', len(unique_groups))
    colors = [cmap(i) for i in range(len(unique_groups))]
    
    for group, color in zip(unique_groups, colors):
        grp_data = df[df['grp'] == group]
        plt.scatter(grp_data['x'], grp_data['y'], label=group, color=color)

def get_colors(df):
    grps = list(set(df['grp']))
    colors = []
    color_palette = ['cyan','yellow','greenyellow','coral','plum']
    colormap = {value: color for value, color in zip(grps, color_palette[:len(grps)])}

    for value in df['grp']:
        colors.append(colormap[value])
    return colors

def plot_2d_tree(colors, G, mst):
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=False, node_size=10, node_color=colors, edge_color='gray', alpha=0.6)
    nx.draw(mst, pos, with_labels=False, node_size=10, node_color=colors, edge_color='r', alpha=1)
    plt.show()