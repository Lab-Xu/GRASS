import numpy
import torch
# from utils import construct_graph, set_params, evaluate
from GRASS_ST.module import GRASS_Integration
import warnings
import datetime
import pickle as pkl
import os
import random
from sklearn.decomposition import PCA
import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import ot

def refine_label_knn(refine_slice_df, n_neighbors=50):
    import sklearn.neighbors
    new_type = []
    coor = refine_slice_df[['imagerow', 'imagecol']]
    # print("coor:", coor)

    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors).fit(coor)
    distances, indices = nbrs.kneighbors(coor, return_distance=True)
    # print("indices shape:", indices.shape)
    n_cell = indices.shape[0]
    for it in range(n_cell):
        neigh_type = [refine_slice_df['old_type'][i] for index, i in enumerate(indices[it]) if index != 0]
        # print('neigh_type:', neigh_type)
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(int(max_type))

    # new_type = np.array(new_type, dtype=int)
    new_type = [str(i) for i in list(new_type)]

    return new_type


def refine_label_pot(refine_slice_df, n_neighbors=50):
    new_type = []
    coor = np.array(refine_slice_df[['imagerow', 'imagecol']])

    distance = ot.dist(coor, coor, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neighbors + 1):
            neigh_type.append(refine_slice_df['old_type'][index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]

    return new_type


def refine_label(adata, refine_method='pot', key='louvain', n_neighbors=50):
    data = {'index': adata.obs.index,
            'imagerow': adata.obsm['spatial'][:, 0],
            'imagecol': adata.obsm['spatial'][:, 1],
            'old_type': adata.obs[key].values,
            'new_type': None,
            'slice_name': adata.obs['slice_name'],
            }

    refine_df = pd.DataFrame(data)

    for slice_name in np.unique(refine_df['slice_name']):
        refine_slice_df = refine_df[refine_df['slice_name'] == slice_name]
        if refine_method == 'pot':
            new_type = refine_label_pot(refine_slice_df, n_neighbors=n_neighbors)
        elif refine_method == 'knn':
            new_type = refine_label_knn(refine_slice_df, n_neighbors=n_neighbors)
        refine_df.loc[refine_df['slice_name'] == slice_name, 'new_type'] = new_type
    return refine_df['new_type']


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm=None, random_seed=2024):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    if used_obsm:
        cluster_data = adata.obsm[used_obsm]
    else:
        cluster_data = adata.X
    # print("cluster_data:", cluster_data)
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(cluster_data), num_cluster, modelNames)

    mclust_res = np.array(res[-2])

    return mclust_res


def res_search_fixed_clus(adata, fixed_clus_count, cluster_method, increment=0.02, seed=2024):
    '''
        arg1(adata)[AnnData matrix]
        arg2(fixed_clus_count)[int]

        return:
            resolution[int]
    '''
    for res in sorted(list(np.arange(0.01, 2.5, increment)), reverse=True):
        sc.tl.louvain(adata, resolution=res, random_state=seed)
        count_unique_leiden = len(pd.DataFrame(adata.obs[cluster_method])[cluster_method].unique())
        if count_unique_leiden <= fixed_clus_count:
            # print("predict cluster count:", count_unique_leiden)
            break
    print("best resolution:", res)
    return res

class Emb_Clustering(object):
    def __init__(
        self,
        st_name_list, 
        graph_dict,
        batch_key = None,
        device: str = None, 
        nb_epochs: int = 1000,
        patience: int = 20,
        lr: int = 0.0008,
        l2_coef: int = 0,
    ):
        
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device =  torch.device('cuda:0') if torch.cuda.is_available() else  torch.device('cpu')
        print("use device:", self.device)

        self.st_name_list = st_name_list
        self.graph_dict = graph_dict
        self.batch_key = batch_key
        self.nb_epochs = nb_epochs
        self.patience = patience
        self.lr = lr
        self.l2_coef = l2_coef
        self.loss_list = []

        self.nei_index_list = []
        self.mps_list = []
        self.pos_list = []
        self.shuf_feats = []
        self.lbl_list = []

        for slice_index, slice_name in enumerate(self.st_name_list):
            temp_dict = self.graph_dict[slice_name]

            if slice_index==0:
                self.feats = temp_dict['feats']
            
            self.nei_index_list.append(temp_dict['nei_index'])
            self.mps_list.append(temp_dict['mps'])
            self.pos_list.append(temp_dict['pos'])
        
        for feat in self.feats:
            nb_nodes = feat.shape[0]
            idx = np.random.permutation(nb_nodes)
            shuf_fts = feat[idx, :]
            self.shuf_feats.append(shuf_fts)

            lbl_1 = torch.ones(1, nb_nodes)
            lbl_2 = torch.zeros(1, nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1)
            self.lbl_list.append(lbl)


    def run_algorithm(self, 
                      hidden_dim_list=[64], feat_drop=0.3, attn_drop=0.5, 
                      sample_rate=[6, 6], nei_num=2, 
                      tau=0.8, lam=0.5, 
                      w_intra=1, w_re=1, w_inter=1,
                      use_mix_expert=False, w_neg=0.5,
                      num_samples=None, num_samples_inter=None):

        feat_dim = self.feats[0].shape[1]
        mapping_dim = feat_dim
        P = int(len(self.mps_list[0]))
        # print("run algorithm device", self.device)

        self.model = GRASS_Integration(hidden_dim_list, feat_dim, mapping_dim, feat_drop, attn_drop,
                     P, sample_rate, nei_num, tau, lam, 
                     use_mix_expert=use_mix_expert, w_neg=w_neg,
                     device=self.device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_coef)
        
        self.model.to(self.device)
        self.feats = [feat.to(self.device) for feat in self.feats]
        self.shuf_feats = [shuf_feat.to(self.device) for shuf_feat in self.shuf_feats]
        self.lbl_list = [lbl.to(self.device) for lbl in self.lbl_list]

        # TODO: 对新的数据加入cuda.
        for i in range(len(self.st_name_list)):
            # self.nei_index_list[i] = [ni.to(self.device) for ni in self.nei_index_list[i]]
            self.mps_list[i] = [mp.to(self.device) for mp in self.mps_list[i]]
            self.pos_list[i] = self.pos_list[i].to(self.device)

        cnt_wait = 0
        best = 1e9
        best_t = 0

        starttime = datetime.datetime.now()
        for epoch in range(self.nb_epochs):
            self.model.train()
            optimiser.zero_grad()
            loss = self.model(self.feats, self.shuf_feats, self.lbl_list,
                              self.pos_list, 
                              self.mps_list, self.nei_index_list,
                              w_intra=w_intra, w_re=w_re, w_inter=w_inter,
                              num_samples=num_samples,
                              num_samples_inter=num_samples_inter)

            self.loss_list.append(loss)

            if epoch % 50 == 0:
                print("loss ", loss.data.cpu())
            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0

            else:
                cnt_wait += 1

            if cnt_wait == self.patience:
                print('Early stopping!')
                break
            loss.backward()
            optimiser.step()
    
    def get_embedding(self, adata, st_list):
        self.model.eval()
        embeds, emb_list, reXs, reX_list = self.model.get_embeds(self.feats, self.mps_list)
        for emb, reX, st_ad in zip(emb_list, reX_list, st_list):
            st_ad.obsm['embed'] = emb.detach().cpu().numpy()
            st_ad.obsm['reX'] = reX.detach().cpu().numpy()
        adata.obsm['embeds'] = embeds.detach().cpu().numpy()
        adata.obsm['reXs'] = reXs.detach().cpu().numpy()
    

    def identify_spatial_domain(self, adata, 
                                cluster_name='mclust',
                                cluster_method='mclust',
                                cluster_num=None,
                                n_neighbors=15, default_resolution=1.5,
                                use_pca=False, n_components=20,
                                use_remove_batch=False,
                                batch_key='slice_name',
                                refinement=True, 
                                refine_method='pot',
                                refine_name=None,
                                n_neighbors_refine=50,
                                seed=2024):
        
        print(f'=====Use {cluster_name} to identify spatial domain=====')

        if cluster_name is None:
            cluster_name = cluster_method

        if refine_name is None:
            cluster_refine_name = f'{cluster_method}_refine_{refine_method}'
        else:
            cluster_refine_name = refine_name

        if cluster_name not in adata.obs.keys():
            if use_remove_batch:
                print("use harmony to remove batch effeat")
                import scanpy.external as sce
                obs = pd.DataFrame({batch_key: adata.obs[batch_key]})
                adataNew = ad.AnnData(X=adata.obsm['embeds'], obs=obs)
                sc.pp.pca(adataNew, n_comps=n_components, random_state=seed)
                sce.pp.harmony_integrate(adataNew, key=[batch_key])
                adata.obsm['embeds_harmony'] = adataNew.obsm['X_pca_harmony']
                clustering_input = adata.obsm['embeds_harmony']
            else:
                if use_pca:
                    pca = PCA(n_components=n_components, random_state=2024)
                    clustering_input = pca.fit_transform(adata.obsm['embeds'].copy())
                    adata.obsm['emb_pca'] = clustering_input
                else:
                    clustering_input = adata.obsm['embeds']

            print("cluster shape:", clustering_input.shape)
            adataNew = ad.AnnData(clustering_input)

            if cluster_method == 'louvain':
                sc.pp.neighbors(adataNew, n_neighbors=n_neighbors, use_rep='X')
                if cluster_num is not None:
                    eval_resolution = res_search_fixed_clus(adataNew, cluster_num, cluster_method)
                else:
                    eval_resolution = default_resolution
                sc.tl.louvain(adataNew, resolution=eval_resolution, key_added=cluster_method, random_state=2024)

                y_pre = np.array(adataNew.obs[cluster_method], dtype=int)

            elif cluster_method == 'mclust':
                if cluster_num is not None:
                    pass
                else:
                    sc.pp.neighbors(adataNew, n_neighbors=n_neighbors, use_rep='X')
                    sc.tl.louvain(adataNew, resolution=default_resolution, key_added='pre_cluster',
                                  random_state=2024)
                    y_pre = np.array(adataNew.obs['pre_cluster'], dtype=int)
                    cluster_num = len(np.unique(y_pre))
                try:
                    y_pre = mclust_R(adataNew, num_cluster=cluster_num)
                except TypeError as e:
                    print(f"An error occurred: {e}")
                    y_pre = 1

            adata.obs[cluster_name] = y_pre
            adata.obs[cluster_name] = adata.obs[cluster_name].astype('int')
            adata.obs[cluster_name] = adata.obs[cluster_name].astype('category')
        else:
            print(f"{cluster_name} results have been obtained...")

        if refinement:
            new_type = refine_label(adata, refine_method=refine_method,
                                    key=cluster_name, n_neighbors=n_neighbors_refine)

            adata.obs[cluster_refine_name] = new_type
            adata.obs[cluster_refine_name] = adata.obs[cluster_refine_name].astype('int')
            adata.obs[cluster_refine_name] = adata.obs[cluster_refine_name].astype('category')
