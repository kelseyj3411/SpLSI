import sys
import os
import time
import pickle
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

# !git clone https://github.com/dx-li/pycvxcluster.git
sys.path.append("./pycvxcluster/")
import pycvxcluster.pycvxcluster

import logging
logging.captureWarnings(True)

from SpLSI.utils import *
from utils.data_helpers import *
from SpLSI import splsi_


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

    return X, edge_df_, coord_df, weights, n, nodes


if __name__ == "__main__":
    K = int(sys.argv[1])
    lamb_start = float(sys.argv[2])
    step_size = float(sys.argv[3])
    grid_len = int(sys.argv[4])
    eps = float(sys.argv[5])

    root_path = os.path.join(os.getcwd(), "data/stanford-crc")
    dataset_root = os.path.join(root_path, "output")
    model_root = os.path.join(root_path, "model")

    #filenames = sorted(set(f.split('.')[0] for f in os.listdir(dataset_root)))
    meta_path = os.path.join(root_path, "charville_labels.csv")
    meta = pd.read_csv(meta_path)
    meta_sub = meta[~pd.isna(meta['primary_outcome'])]
    filenames = meta_sub['region_id'].to_list()
    

    D_all = pd.DataFrame()
    edge_all = pd.DataFrame()
    coords_all = pd.DataFrame()

    s = 0
    #filenames_s = random.sample(filenames, 100)
    for filename in filenames:
            paths = {kind: os.path.join(dataset_root, f"{filename}.{kind}.csv") for kind in ['D', 'edge', 'coord', 'type', 'model']}
            D = pd.read_csv(paths['D'], index_col=0, converters={0: tuple_converter})
            D['filename'] = filename 
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

            D = D[D.iloc[:, :-1].sum(axis=1) >= 10]
            idx = [y for x,y in D.index]
            edge_df = edge_df[(edge_df['src'].isin(idx)) & (edge_df['tgt'].isin(idx))]
            coords_df = coords_df[coords_df['CELL_ID'].isin(idx)]

            D_all = pd.concat([D_all, D], axis=0, ignore_index=False)
            edge_all = pd.concat([edge_all, edge_df], axis=0, ignore_index=True)
            coords_all = pd.concat([coords_all, coords_df], axis=0, ignore_index=True)
            s+= D.shape[0]
    print(f"D has {s} rows.")

    patient_id_path = os.path.join(root_path, "selected_patient_id.csv")
    if not os.path.exists(patient_id_path):
          patient_id_df = D_all['filename']
          patient_id_df.to_csv(patient_id_path, index=False)

    D_all = D_all.drop(columns=['filename'])
    X, edge_df, coord_df, weights, n, nodes = preprocess_crc(coords_all, edge_all, D_all, phi=0.1)
    start_time = time.time()
    model_splsi = splsi_.SpLSI_(
        lamb_start=lamb_start, step_size=step_size, grid_len=grid_len, verbose=1, eps=eps
    )
    model_splsi.fit(X.values, K, edge_df, weights)
    time_splsi = time.time() - start_time
    print(time_splsi)

    save_path = os.path.join(model_root, f'model_splsi_all_{K}_.pkl')
    W_hat = model_splsi.W_hat
    A_hat = model_splsi.A_hat
    csv_path_name_A = os.path.join(model_root, f'A_hat_splsi_{K}_.csv')
    np.savetxt(csv_path_name_A, A_hat, delimiter=',')
    csv_path_name_W = os.path.join(model_root, f'W_hat_splsi_{K}_.csv')
    np.savetxt(csv_path_name_W, W_hat, delimiter=',')

    with open(save_path, "wb") as f:
            pickle.dump(model_splsi, f)
