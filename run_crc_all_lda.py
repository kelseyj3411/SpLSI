from sklearn.decomposition import LatentDirichletAllocation

import sys
import os
import time
import pickle
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

import logging
logging.captureWarnings(True)

from SpLSI.utils import *
from utils.data_helpers import *
from SpLSI import splsi_


if __name__ == "__main__":
    K = int(sys.argv[1])

    root_path = os.path.join(os.getcwd(), "data/stanford-crc")
    dataset_root = os.path.join(root_path, "output")
    model_root = os.path.join(root_path, "model")

    meta_path = os.path.join(root_path, "charville_labels.csv")
    meta = pd.read_csv(meta_path)
    meta_sub = meta[~pd.isna(meta['primary_outcome'])]
    filenames = meta_sub['region_id'].to_list()
    print(len(filenames))

    D_all = pd.DataFrame()

    s = 0
    for filename in filenames:
            paths = {kind: os.path.join(dataset_root, f"{filename}.{kind}.csv") for kind in ['D', 'edge', 'coord', 'type', 'model']}
            D = pd.read_csv(paths['D'], index_col=0, converters={0: tuple_converter})
            
            coord_df = pd.read_csv(paths['coord'], index_col=0)
            cell_to_idx_dict = dict(zip(coord_df["CELL_ID"], [i+s for i in range(coord_df.shape[0])]))

            new_index = [(x,cell_to_idx_dict[y]) for x,y in D.index]
            D.index = new_index

            D = D[D.sum(axis=1)>=10]
            
            D_all = pd.concat([D_all, D], axis=0, ignore_index=False)
            s+= D.shape[0]

    print(f"D has {s} rows.")
    start_time = time.time()
    lda = LatentDirichletAllocation(n_components=K, random_state=0)
    lda.fit(D_all.values)
    time_lda = time.time() - start_time
    print(time_lda)

    save_path = os.path.join(model_root, f'model_lda_{K}.pkl')
    W_hat = lda.transform(D_all.values)
    A_hat = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    csv_path_name_A = os.path.join(model_root, f'A_hat_lda_{K}.csv')
    np.savetxt(csv_path_name_A, A_hat, delimiter=',')
    csv_path_name_W = os.path.join(model_root, f'W_hat_lda_{K}.csv')
    np.savetxt(csv_path_name_W, W_hat, delimiter=',')

    with open(save_path, "wb") as f:
            pickle.dump(lda, f)
