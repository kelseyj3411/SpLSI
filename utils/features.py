import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import palettable.cartocolors.qualitative as qual_palettes
import pandas as pd
import seaborn as sns
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy.special import digamma
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm
from collections import defaultdict


def plot_bcell_topic_multicolor(
    ax, sample_idx, topic_weights, spleen_dfs, name_list, lamb
):
    color_palette = sns.color_palette("bright", 10)
    # color_palette = sns.color_palette("hls", 10)
    colors = np.array(color_palette[: topic_weights.shape[0]])
    cell_coords = spleen_dfs[sample_idx]
    non_b_coords = cell_coords[~cell_coords.isb]
    ax.scatter(
        non_b_coords["sample.Y"],
        non_b_coords["sample.X"],
        s=1,
        c="k",
        marker="x",
        label="Non-B",
        alpha=0.2,
    )

    cell_indices = [int(eval(name)[1]) for name in name_list]
    coords = spleen_dfs[sample_idx].loc[cell_indices]

    ax.scatter(
        coords["sample.Y"],
        coords["sample.X"],
        s=3,
        c=colors[np.argmax(np.array(topic_weights), axis=0), :],
    )

    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.set_title(f"Spatial Penalty= {lamb}")


def get_component_mapping(stats_1, stats_2):
    similarity = stats_1 @ stats_2.T
    assignment = linear_sum_assignment(-similarity)
    mapping = {k: v for k, v in zip(*assignment)}
    return mapping


def get_consistent_orders(stats_list):
    d = stats_list[0].shape[1]
    n_topics = [stats.shape[0] for stats in stats_list]
    assert all([stats.shape[1] == d for stats in stats_list])
    assert all([n1 <= n2 for n1, n2 in zip(n_topics[:-1], n_topics[1:])])
    orders = [list(range(n_topics[0]))]
    for stats_1, stats_2 in zip(stats_list[:-1], stats_list[1:]):
        n_topics_1 = stats_1.shape[0]
        n_topics_2 = stats_2.shape[0]
        mapping = get_component_mapping(stats_1[orders[-1], :], stats_2)
        mapped = mapping.values()
        unmapped = set(range(n_topics_2)).difference(mapped)
        order = [mapping[k] for k in range(n_topics_1)] + list(unmapped)
        orders.append(order)
    return orders


def apply_order(weights_df, n_topics):
    stats_list = [weights for weights in weights_df.values()]
    orders = get_consistent_orders(stats_list)
    for i, n_topic in enumerate(n_topics):
        arr = weights_df[n_topic]
        order_list = orders[i]
        weights_df[n_topic] = arr[order_list]


def calculate_sum_of_squared_neighborhood_weights(edge_list):
    neighborhood_weights = defaultdict(int)
    for i, j, weight in edge_list:
        neighborhood_weights[i] += weight
        neighborhood_weights[j] += weight

    sum_squared_weights = 0
    for node, weights in neighborhood_weights.items():
        squared_sum = weights**2
        sum_squared_weights += squared_sum

    return sum_squared_weights


def moran(weights_df, edge):
    weights = 1 / edge[:, -1]
    weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
    s0 = np.sum(weights)
    vals = np.argmax(np.array(weights_df), axis=0)
    vals = vals - np.mean(vals)
    n = vals.shape[0]
    node_i = edge[:, 0].astype(int)
    node_j = edge[:, 1].astype(int)
    num = n * (np.sum(weights * vals[node_i] * vals[node_j]))
    denom = s0 * np.sum(vals**2)
    I = num / denom
    # s1 = 2*np.sum(weights**2)
    # s2 = calculate_sum_of_squared_neighborhood_weights(edge.tolist())
    # EI = -1/(n-1)
    # D = np.sum(vals**4)/(np.sum(vals**2))**2
    # A = n*((n**2-3*n+3)*s1-n*s2+3*(s0**2))
    # B = D*((n**2-n)*s1-2*n*s2+6*(s0**2))
    # C = (n-1)*(n-2)*(n-3)*(s0**2)
    # EI2 = (A-B)/C
    # V = EI2-(EI**2)
    # Z = (I-EI)/np.sqrt(V)
    return I


def get_CHAOS(weights_df, edge):
    d_ij = 0
    n_topics, n_cells = weights_df.shape
    src_nodes = edge[:, 0]
    dst_nodes = edge[:, 1]
    edge_lengths = edge[:, 2]
    for k in range(n_topics):
        cell_idx = np.arange(n_cells)
        K_cell_idx = cell_idx[np.argmax(np.array(weights_df), axis=0) == k]
        src_indices = np.isin(cell_idx[src_nodes.astype(int)], K_cell_idx)
        dst_indices = np.isin(cell_idx[dst_nodes.astype(int)], K_cell_idx)
        d_ijk = np.sum(edge_lengths[src_indices & dst_indices])
        d_ij += d_ijk
    d_ij = d_ij / n_cells
    return d_ij


def get_PAS(weights_df, edge):
    n_topics, n_cells = weights_df.shape
    src_nodes = edge[:, 0]
    dst_nodes = edge[:, 1]
    cell_idx = np.arange(n_cells)
    topics = np.argmax(np.array(weights_df), axis=0)
    cell_topics = pd.DataFrame({"cell": cell_idx, "topic": topics})
    edge_sets = pd.DataFrame(
        {"src": cell_idx[src_nodes.astype(int)], "dst": cell_idx[dst_nodes.astype(int)]}
    )

    joined_df = edge_sets.merge(cell_topics, left_on="src", right_on="cell").merge(
        cell_topics, left_on="dst", right_on="cell", suffixes=("_src", "_dst")
    )
    cell_counts = joined_df.groupby("src").apply(
        lambda x: (x["topic_src"] != x["topic_dst"]).mean()
    )
    count = (cell_counts >= 0.6).sum()
    ratio = count / cell_idx.shape
    return ratio


def plot_score(ax, list2, list3, n_topics, s_name, sample):
    # ax.plot(n_topics, list1, linestyle=':', color='lightsteelblue', marker='o',label="LDA")
    ax.plot(n_topics, list2, linestyle=":", color="blue", marker="o", label="SLDA")
    ax.plot(n_topics, list3, linestyle=":", color="red", marker="o", label="TopicScore")
    ax.set_xlabel("Number of Topics")
    ax.axes.set_title(f"{s_name} score for {sample}")
    ax.legend()


def get_avg_corr(weights_df1, weights_df2, n_topics):
    avg_corr = []
    stats_list_SLDA = [weights for weights in weights_df1.values()]
    stats_list_tscore = [weights for weights in weights_df2.values()]
    for i, n_topic in enumerate(n_topics):
        order = list(
            get_component_mapping(stats_list_SLDA[i], stats_list_tscore[i]).values()
        )
        tscore = weights_df2[n_topic]
        tscore_rearranged = tscore[order]
        slda = weights_df1[n_topic]
        corr = np.abs(
            [
                np.corrcoef(slda[:, j], tscore_rearranged[:, j])[0, 1]
                for j in range(slda.shape[1])
            ]
        )
        avg_corr.append(np.mean(corr))
    return avg_corr
