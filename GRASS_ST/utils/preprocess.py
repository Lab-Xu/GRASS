import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp


def preprocess(st_list, 
               st_name_list,
               have_annotation=False,
               batch_key='slice_name',
               n_top_genes=3000,
               filter_genes=False,
               integ_method='A',
               detect_NaN=False,
               ):
    
    if integ_method=='A':
        st_adata_concat = ad.concat(st_list, label=batch_key, keys=st_name_list)
        if have_annotation:
            st_adata_concat.obs['ground_truth'] = st_adata_concat.obs['ground_truth'].astype('category')
        st_adata_concat.obs[batch_key] = st_adata_concat.obs[batch_key].astype('category')

        if filter_genes:
            sc.pp.filter_genes(st_adata_concat, min_counts=3)
        sc.pp.highly_variable_genes(st_adata_concat, flavor="seurat_v3", n_top_genes=n_top_genes)
        sc.pp.normalize_total(st_adata_concat, target_sum=1e4)
        sc.pp.log1p(st_adata_concat)
        sc.pp.scale(st_adata_concat, zero_center=False, max_value=10)
        st_adata_concat = st_adata_concat[:, st_adata_concat.var['highly_variable']]

    elif integ_method=='B':
        st_adata_list = []
        for st_adata in st_list:
            sc.pp.highly_variable_genes(st_adata, flavor="seurat_v3", n_top_genes=n_top_genes)
            sc.pp.normalize_total(st_adata, target_sum=1e4)
            sc.pp.log1p(st_adata)
            sc.pp.scale(st_adata, zero_center=False, max_value=10)
            adata_hv = st_adata[:, st_adata.var['highly_variable']]
            st_adata_list.append(adata_hv)

        st_adata_concat = ad.concat(st_adata_list, label=batch_key, keys=st_name_list)
        st_adata_concat.obs['ground_truth'] = st_adata_concat.obs['ground_truth'].astype('category')
        st_adata_concat.obs["batch_name"] = st_adata_concat.obs["slice_name"].astype('category')
    
    if detect_NaN:
        if isinstance(st_adata_concat.X, np.ndarray):
            mask = ~np.isnan(st_adata_concat.X).any(axis=0)
            st_adata_concat = st_adata_concat[:, mask]
        else:
            dense_X = st_adata_concat.X.toarray()
            mask = ~np.isnan(dense_X).any(axis=0)
            st_adata_concat = st_adata_concat[:, mask]

    for index in range(len(st_name_list)):
        integX = st_adata_concat[st_adata_concat.obs[batch_key]==st_name_list[index]].X
        st_list[index].obsm['integX'] = integX

    st_adata_concat.obsm['counts'] = st_adata_concat.X.copy()

    return st_adata_concat