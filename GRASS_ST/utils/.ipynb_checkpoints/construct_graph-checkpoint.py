import numpy as np
import pandas as pd
import random
import torch
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from .graph import get_data_list
import itertools
import numpy as np
from joblib import Parallel, delayed
import time

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def process_slice(slice_name, nei_index_dict, nei_graph_dict, feats_dict, threshold, weight_graph=False):
    graph_data = {'nei_index': [], 'feats': [], 'mps': [], 'pos': None}

    # Process nei_index
    for name, neigh_array  in nei_index_dict.items():
        if (name[0]==slice_name)&(name[1]=='1'):
            graph_data['nei_index'].append([torch.LongTensor(i) for i in neigh_array])
    

    # Process feats
    feat_data = [torch.FloatTensor(preprocess_features(feat)) for (name, feat) in feats_dict.items()]
    graph_data['feats'].extend(feat_data)

    for name, neigh_graph in nei_graph_dict.items():
        if (name==slice_name):
            # Process mps
            if weight_graph:
                A_mps = neigh_graph + neigh_graph.T
            else:
                A_mps = neigh_graph + neigh_graph.T
                A_mps = np.where(A_mps > 0, 1, 0)
            
            # Process pos
            num_neighbors = neigh_array.shape[1]
            min_overlap = int(threshold * num_neighbors)
            neighbor_matrix = neigh_graph
            overlap_matrix = neighbor_matrix @ neighbor_matrix.T
            A_pos = (overlap_matrix > min_overlap).astype(int)

            # Process mps
            A_mps_sparse = sp.coo_matrix(A_mps)
            A_mps_normalized = normalize_adj(A_mps_sparse)
            A_mps_p = sparse_mx_to_torch_sparse_tensor(A_mps_normalized)
            graph_data['mps'].append(A_mps_p)

            # Process pos
            A_pos_sparse = sp.coo_matrix(A_pos)
            A_pos_p = sparse_mx_to_torch_sparse_tensor(A_pos_sparse)
            graph_data['pos'] = A_pos_p

    # for name, neigh_array in nei_index_dict.items():
    #     if (name[0]==slice_name)&(name[1]=='0'):

    #         N = neigh_array.shape[0]
    #         # Process mps
    #         A_mps = np.zeros((N, N), dtype=int)
    #         # Process pos
    #         num_neighbors = neigh_array.shape[1]
    #         min_overlap = int(threshold * num_neighbors)
    #         neighbor_matrix = np.zeros((N, N), dtype=int)

    #         for i, neighbors in enumerate(neigh_array):
    #             # Process mps
    #             A_mps[i, neighbors] = 1
    #             A_mps[neighbors, i] = 1
    #             # Process pos
    #             neighbors = np.array(neighbors, dtype=int)  # Ensure neighbors are integers
    #             neighbor_matrix[i, neighbors] = 1  # Mark neighbor relations
    #         # Process pos
    #         overlap_matrix = neighbor_matrix @ neighbor_matrix.T
    #         A_pos = (overlap_matrix > min_overlap).astype(int)

    #         # Process mps
    #         A_mps_sparse = sp.coo_matrix(A_mps)
    #         A_mps_normalized = normalize_adj(A_mps_sparse)
    #         A_mps_p = sparse_mx_to_torch_sparse_tensor(A_mps_normalized)
    #         graph_data['mps'].append(A_mps_p)

    #         # Process pos
    #         A_pos_sparse = sp.coo_matrix(A_pos)
    #         A_pos_p = sparse_mx_to_torch_sparse_tensor(A_pos_sparse)
    #         graph_data['pos'] = A_pos_p

    return slice_name, graph_data

def construct_graph(adata, st_name_list,
                    n_neigh=10,
                    threshold=0.6,
                    tao=0.5,
                    latent_key='X_batch_removal', 
                    input_data_key='counts',
                    spatial_key='spatial',
                    batch_key='slice_name',
                    weight_graph=False,
                    cluster_num=None,
                    device=None,
                    ):
    """
    Initialize G2 model.

    Given datasets and parameters to initialize Spoint model.

    Parameters
    ----------
    st_ad_list
        An AnnData object List representing spatial transcriptomic datasets.
    n_top_genes
        Number of highly-variable genes to keep.

    Return
    -------
    G2_model
        G2Model Object
        :param params:
        :param augment_graph:
        :param use_gpu:
        :param seed:
        :param n_top_genes:
        :param st_name_list:
        :param st_list:

    """

    if device is not None:
        device = torch.device(device)
    else:
        device =  torch.device('cuda:0') if torch.cuda.is_available() else  torch.device('cpu')
    print("construct graph use device:", device)

    time_1 = time.time()

    nei_index_dict, nei_graph_dict, feats_dict = get_data_list(
        adata, 
        st_name_list,
        tao=tao,
        n_neigh=n_neigh,
        latent_key=latent_key,
        input_data_key=input_data_key,
        spatial_key=spatial_key,
        batch_key=batch_key,
        cluster_num=cluster_num,
        device=device)
    # print("nei_graph_dict:", nei_graph_dict)
    
    time_2 = time.time()

    results = Parallel(n_jobs=-1)(
            delayed(process_slice)(slice_name, nei_index_dict, nei_graph_dict, feats_dict, threshold, weight_graph) 
            for slice_name in st_name_list
        )
    graph_dict = dict(results)

    time_3 = time.time()
    elapsed_time1 = time_2 - time_1
    print(f"get data_list time: {elapsed_time1:.6f}")
    elapsed_time2 = time_3 - time_2
    print(f"construct graph_dict time: {elapsed_time2:.6f}")
    
    return graph_dict
