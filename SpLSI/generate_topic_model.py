import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
import networkx as nx
#from netgraph import Graph
#from netgraph import get_sugiyama_layout


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
    np.random.seed(127)
    coords = np.zeros((n, 2))
    coords[:, 0] = np.random.uniform(-3, 3, n)
    coords[:, 1] = np.random.uniform(-3, 3, n)

    cluster_obj = KMeans(n_clusters=20, init=coords[get_initial_centers(coords, 20), :], n_init=1)
    grps = cluster_obj.fit_predict(coords)

    df = pd.DataFrame(coords, columns=['x','y'])
    df['grp'] = grps % K
    return df

def generate_W(df, N, n, p, K, r):
    W = np.zeros((K, n))
    for k in range(K):
        alpha = np.random.uniform(0.1, 0.3, K)
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

def generate_graph_from_weights(df, weights, n):
    np.random.seed(127)
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

def generate_mst(df, n, K, nearest_n, phi):
    weights = generate_weights(df, K, nearest_n, phi)
    G = generate_graph_from_weights(df, weights, n)
    mst, path = get_mst_path(G)
    return G, mst, path, weights

def get_parent_node(path, mst, srn, nodenum):
    neighs = list(mst.adj[nodenum].keys())
    length_to_srn = [path[neigh][srn] for neigh in neighs]
    parent = neighs[np.argmin(length_to_srn)]
    return parent

def get_folds(mst, path, n, plot_tree=True):
    np.random.seed(127)
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
    len1 = len(fold1)
    len2 = len(fold2)
    print(f"Fold1 size is {len1}")
    print(f"Fold2 size is {len2}")
    if plot_tree:
        nx.draw_kamada_kawai(mst, node_color = colors, node_size=10)
    return srn, fold1, fold2

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