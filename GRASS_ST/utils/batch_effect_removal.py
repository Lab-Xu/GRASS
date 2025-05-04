import scipy.sparse as sp
import numpy as np
import anndata as ad
import scanpy as sc
import scanpy.external as sce
import pandas as pd
from scipy import sparse
from torch_geometric.data import Data, DataLoader
from sklearn.decomposition import PCA
import ot
import os
import random
import torch
import torch.backends.cudnn as cudnn
from . import batch_effect_removal_AE


def batch_effect_removal(adata_concat, 
                         method_type='harmony', 
                         batch_key='slice_name',
                         latent_key='X_batch_removal',
                         n_comps=200, 
                        #  n_latent=32,
                         epoch_ae=100, 
                         seed=2024,
                         device=None):
    # assert method_type.lower() in ['pca', 'harmony', 'scanorama', 'scVI', 'scdiffusion', None]

    # Feature
    datas = []
    if method_type:
        print(f'=====Use {method_type} to remove batch effect=====')
    else:
        print("use original data")

    if isinstance(adata_concat.X, np.ndarray):
        data_mat = adata_concat.X.copy()
    else:
        data_mat = adata_concat.X.todense().copy()
    

    if method_type is not None:
        if method_type.lower() == 'harmony':
            sc.pp.pca(adata_concat, n_comps=n_comps, random_state=seed)
            sce.pp.harmony_integrate(adata_concat, key=[batch_key])

            X = adata_concat.obsm['X_pca_harmony']

        elif method_type.lower() == 'scanorama':
            sc.pp.pca(adata_concat, n_comps=n_comps, random_state=seed)
            sce.pp.scanorama_integrate(adata_concat, batch_key, verbose=1)

            X = adata_concat.obsm['X_scanorama']

        elif method_type.lower() == 'pca':
            # sc.pp.pca(adata_concat, n_comps=n_comps, random_state=seed)
            # pca = PCA(n_components=n_comps, random_state=seed)
            # input_data = pca.fit_transform(adata_concat.X.copy())
            # adata_concat.obsm['X_pca'] = input_data
            sc.tl.pca(adata_concat, svd_solver='auto')
            X = adata_concat.obsm['X_pca'][:,:n_comps]

            # X = adata_concat.obsm['X_pca']

        elif method_type.lower() == 'ae':
            ae_model = batch_effect_removal_AE.AE_Model(adata_concat,
                                                        batch_key=batch_key,
                                                        n_latent=n_comps,
                                                        likelihood='zinb',
                                                        result_path='OB',
                                                        device=device,)
            ae_model.run_algorithm(epoch_ae=epoch_ae)
            ae_model.get_latent()
            X = adata_concat.obsm['latent']
            print("run over")
    else:
        X = data_mat
    # print("X shape:{}".format(X.shape))
    
    # 转换为系数矩阵
    adata_concat.obsm[latent_key] = sparse.csr_matrix(X)

    # print('adata_concat:', adata_concat)

    # if isinstance(adata_concat.X, np.ndarray):
    #     adata_concat.obsm[latent_key] = X
    # else:
    #     adata_concat.obsm[latent_key] = sparse.csr_matrix(X)