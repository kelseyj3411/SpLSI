import sys
import os
import gc
import pickle
import numpy as np
import pandas as pd
from itertools import combinations

from scipy.sparse import csr_matrix
from collections import defaultdict

# !git clone https://github.com/dx-li/pycvxcluster.git
sys.path.append("./pycvxcluster/")
import pycvxcluster.pycvxcluster

import logging
logging.captureWarnings(True)

from SpLSI.utils import *
from utils.data_helpers import *
from SpLSI import splsi_

from mpi4py import MPI

def preprocess_crc(coord_df, edge_df, D, phi):
    new_columns = [col.replace("X", "x").replace("Y", "y") for col in coord_df.columns]
    coord_df.columns = new_columns

    # get cell index
    cell_to_idx_dict = dict(zip(coord_df["CELL_ID"], range(coord_df.shape[0])))

    # normalize coordinate to (0,1)
    coord_df[["x", "y"]] = normaliza_coords(coord_df)

    # get weight
    edge_df_ = edge_df.copy()
    edge_df_["src"] = edge_df["src"].map(cell_to_idx_dict)
    edge_df_["tgt"] = edge_df["tgt"].map(cell_to_idx_dict)
    edge_df_["weight"] = dist_to_exp_weight(edge_df_, coord_df, phi)

    # edge, coord, X, weights
    nodes = coord_df.index.tolist()
    row_sums = D.sum(axis=1)
    X = D.div(row_sums, axis=0)  # normalize
    n = X.shape[0]
    weights = csr_matrix(
        (edge_df_["weight"].values, (edge_df_["src"].values, edge_df_["tgt"].values)),
        shape=(n, n),
    )

    return X.values, edge_df_, coord_df, weights, n, nodes

def divide_folds(filenames, num_parts=2):
    np.random.shuffle(filenames)
    avg_length = len(filenames) / float(num_parts)

    divided_folds = []
    last = 0.0

    while last < len(filenames):
        divided_folds.append(filenames[int(last):int(last + avg_length)])
        last += avg_length

    return divided_folds


def shuffle_folds(filenames, dataset_root, nfolds):
    data_inputs = []

    divided_folds = divide_folds(filenames, nfolds)

    for i in range(nfolds):
        D_fold = pd.DataFrame()
        edge_fold = pd.DataFrame()
        coords_fold = pd.DataFrame()

        set = divided_folds[i]
        s = 0
        for filename in set:
            paths = {kind: os.path.join(dataset_root, f"{filename}.{kind}.csv") for kind in ['D', 'edge', 'coord', 'type', 'model']}
            D = pd.read_csv(paths['D'], index_col=0, converters={0: tuple_converter})
            edge_df = pd.read_csv(paths['edge'], index_col=0)
            coord_df = pd.read_csv(paths['coord'], index_col=0)
            type_df = pd.read_csv(paths['type'], index_col=0)
            coords_df = pd.merge(coord_df, type_df).reset_index(drop=True)

            cell_to_idx_dict = dict(zip(coord_df["CELL_ID"], [i+s for i in range(coord_df.shape[0])]))

            edge_df["src"] = edge_df["src"].map(cell_to_idx_dict)
            edge_df["tgt"] = edge_df["tgt"].map(cell_to_idx_dict)
            coords_df["CELL_ID"] = coords_df["CELL_ID"].map(cell_to_idx_dict)
            new_index = [(x,cell_to_idx_dict[y]) for x,y in D.index]
            D.index = new_index

            D = D[D.sum(axis=1)>=10]
            idx = [y for x,y in D.index]
            edge_df = edge_df[(edge_df['src'].isin(idx)) & (edge_df['tgt'].isin(idx))]
            coords_df = coords_df[coords_df['CELL_ID'].isin(idx)]

            D_fold = pd.concat([D_fold, D], axis=0, ignore_index=False)
            edge_fold = pd.concat([edge_fold, edge_df], axis=0, ignore_index=True)
            coords_fold = pd.concat([coords_fold, coords_df], axis=0, ignore_index=True)
            del D, edge_df, coords_df
            gc.collect()

        X, edge_df, _, weights, _, _ = preprocess_crc(coords_fold, edge_fold, D_fold, phi=0.1)
        data_inputs.append((X, edge_df, weights))
        del X, edge_df, weights
        gc.collect()

    return data_inputs

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    size = comm.Get_size() 
    rank = comm.Get_rank() 

    K = int(sys.argv[1])
    lamb_start = float(sys.argv[2])
    step_size = float(sys.argv[3])
    grid_len = int(sys.argv[4])
    eps = float(sys.argv[5])


    if rank == 0:
        print("Processing data...")
        nfolds = 5
        root_path = os.path.join(os.getcwd(), "data/stanford-crc")
        dataset_root = os.path.join(root_path, "output")
        model_root = os.path.join(root_path, "model")

        filenames = sorted(set(f.split('.')[0] for f in os.listdir(dataset_root)))
        filenames = [f for f in filenames if f]
        data_inputs = shuffle_folds(filenames, dataset_root, nfolds)

        X_full = np.vstack([input[0] for input in data_inputs])
        chunks = [[] for _ in range(size)]
        for i, data in enumerate(data_inputs):
            chunks[i % size].append(data)
        del data_inputs
        gc.collect()

    else:
        chunks = None
        X_full = None

    print("Scattering inputs...")
    tasks = comm.scatter(chunks, root=0)

    # Run in each node
    start_time = time.time()
    print(f"Process {rank} calculating alignment score...")

    local_As = []
    for task in tasks:
        X, edge_df, weights = task
        model_splsi = splsi_.SpLSI_(
                lamb_start=lamb_start, step_size=step_size, grid_len=grid_len, verbose=0, eps=eps
            )
        model_splsi.fit(X, K, edge_df, weights)
        local_As.append(model_splsi.A_hat)

    all_Ahats = comm.gather(local_As, root=0)

    if rank == 0:
        results = defaultdict(list)
        for i, j in combinations(range(len(all_Ahats[0])), 2):
            P = get_component_mapping(all_Ahats[i][0], all_Ahats[j][0])
            A_1 = all_Ahats[i][0]
            A_2 = all_Ahats[j][0]
            A_1 = P.T @ A_1
            K = A_1.shape[0]

            # l1 distance
            l1_dist = np.sum(np.abs(A_1 - A_2))
            results['K'].append(K)
            results['l1_dist'].append(l1_dist/K)

            # cosine similarity
            A_1_norm = A_1 / norm(A_1, axis=1, keepdims=True)
            A_2_norm = A_2 / norm(A_2, axis=1, keepdims=True)
            diag_cos = np.mean(np.diag(A_1_norm @ A_2_norm.T))
            results['cos_sim'].append(diag_cos)

            # cosine similarity ratio
            off_cos = (np.sum(A_1_norm @ A_2_norm.T)-np.sum(np.diag(A_1_norm @ A_2_norm.T)))/(K*K - K)
            r = diag_cos/off_cos
            results['cos_sim_ratio'].append(r)
        results_df = pd.DataFrame(results)
        results_df.to_csv('crc_chooseK_results.csv')