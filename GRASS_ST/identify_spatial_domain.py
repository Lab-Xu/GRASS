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

def clustering(adata, 
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
