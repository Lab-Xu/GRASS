import numpy as np
import pandas as pd
import itertools
import hnswlib
import networkx as nx
import anndata as ad


def nn(ds1, ds2, names1, names2, knn=50, metric_p=2):
    from sklearn.neighbors import NearestNeighbors
    # Find nearest neighbors of first dataset.
    nn_ = NearestNeighbors(knn, p=metric_p)
    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match


def nn_approx(ds1, ds2, names1, names2, knn=50):
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=100, M=16)
    p.set_ef(10)
    p.add_items(ds2)
    ind, distances = p.knn_query(ds1, k=knn)
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))
    return match


def mnn(ds1, ds2, names1, names2, knn=20, save_on_disk=True, approx=True):
    if approx:
        # Find nearest neighbors in first direction.
        # output KNN point for each point in ds1.  match1 is a set(): (points in names1, points in names2), the size of the set is ds1.shape[0]*knn
        match1 = nn_approx(ds1, ds2, names1, names2, knn=knn)  # , save_on_disk = save_on_disk)
        # Find nearest neighbors in second direction.
        match2 = nn_approx(ds2, ds1, names2, names1, knn=knn)  # , save_on_disk = save_on_disk)
    else:
        match1 = nn(ds1, ds2, names1, names2, knn=knn)
        match2 = nn(ds2, ds1, names2, names1, knn=knn)
    # Compute mutual nearest neighbors.
    mutual = match1 & set([(b, a) for a, b in match2])

    return mutual


def create_dictionary_mnn(adata, use_rep, batch_name, k=50, save_on_disk=True, approx=True, verbose=1, iter_comb=None):
    cell_names = adata.obs_names

    batch_list = adata.obs[batch_name]
    datasets = []
    datasets_pcs = []
    cells = []
    for i in batch_list.unique():
        datasets.append(adata[batch_list == i])
        datasets_pcs.append(adata[batch_list == i].obsm[use_rep])
        cells.append(cell_names[batch_list == i])

    batch_name_df = pd.DataFrame(np.array(batch_list.unique()))
    mnns = dict()

    if iter_comb is None:
        iter_comb = list(itertools.combinations(range(len(cells)), 2))
    for comb in iter_comb:
        i = comb[0]
        j = comb[1]
        key_name1 = batch_name_df.loc[comb[0]].values[0] + "_" + batch_name_df.loc[comb[1]].values[0]
        mnns[
            key_name1] = {}  # for multiple-slice setting, the key_names1 can avoid the mnns replaced by previous slice-pair
        if (verbose > 0):
            print('Processing datasets {}'.format((i, j)))

        new = list(cells[j])
        ref = list(cells[i])

        ds1 = adata[new].obsm[use_rep]
        ds2 = adata[ref].obsm[use_rep]
        names1 = new
        names2 = ref
        # if k>1，one point in ds1 may have multiple MNN points in ds2.
        match = mnn(ds1, ds2, names1, names2, knn=k, save_on_disk=save_on_disk, approx=approx)

        G = nx.Graph()
        G.add_edges_from(match)
        node_names = np.array(G.nodes)
        anchors = list(node_names)
        adj = nx.adjacency_matrix(G)
        tmp = np.split(adj.indices, adj.indptr[1:-1])

        for i in range(0, len(anchors)):
            key = anchors[i]
            i = tmp[i]
            names = list(node_names[i])
            mnns[key_name1][key] = names
    return (mnns)


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    # assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def ICP_3D_reconstruct(source_data, target_data,
                       source_cluster, target_cluster,
                       matching_dict,
                       cluster_method='mclust',):

    # print("source data shape:", source_data.shape)
    # print("target data shape:", target_data.shape)
    adata_slice1 = target_data[target_data.obs[cluster_method].isin([target_cluster])]
    adata_slice2 = source_data[source_data.obs[cluster_method].isin([source_cluster])]
    source_as_dict = dict(zip(list(target_data.obs_names), range(0, target_data.shape[0])))
    target_as_dict = dict(zip(list(source_data.obs_names), range(0, source_data.shape[0])))

    source_section_as_dict = dict(zip(list(adata_slice1.obs_names), range(0, adata_slice1.shape[0])))
    target_section_as_dict = dict(zip(list(adata_slice2.obs_names), range(0, adata_slice2.shape[0])))

    ## index number
    source_anchor_ind = list(map(lambda _: source_as_dict[_], adata_slice1.obs_names))
    target_anchor_ind = list(map(lambda _: target_as_dict[_], adata_slice2.obs_names))

    key_points_src = []
    key_points_dst = []
    for k, v in matching_dict.items():
        if (k in target_anchor_ind) & (v in source_anchor_ind):
            key_points_dst.append(k)
            key_points_src.append(v)

    # print("matching_dict len:", len(matching_dict))
    MNN_ind_src = list(map(lambda _: source_section_as_dict[_], target_data.obs_names[key_points_src]))
    MNN_ind_dst = list(map(lambda _: target_section_as_dict[_], source_data.obs_names[key_points_dst]))

    # print("MNN_ind_src len:{}, MNN_ind_dst len:{}".format(len(MNN_ind_src), len(MNN_ind_dst)))

    ####### ICP alignment
    init_pose = None
    max_iterations = 100
    tolerance = 0.001

    coor_src = adata_slice1.obsm["spatial"]  ## to_be_aligned
    coor_dst = adata_slice2.obsm["spatial"]  ## reference_points

    coor_used = coor_src  ## Batch_list[1][Batch_list[1].obs['annotation']==2].obsm["spatial"]
    coor_all = target_data.obsm["spatial"].copy()
    coor_used = np.concatenate([coor_used, np.expand_dims(np.ones(coor_used.shape[0]), axis=1)], axis=1).T
    coor_all = np.concatenate([coor_all, np.expand_dims(np.ones(coor_all.shape[0]), axis=1)], axis=1).T
    A = coor_src  ## to_be_aligned
    B = coor_dst  ## reference_points

    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)
    prev_error = 0

    for ii in range(max_iterations + 1):
        p1 = src[:m, MNN_ind_src].T
        p2 = dst[:m, MNN_ind_dst].T
        T, _, _ = best_fit_transform(src[:m, MNN_ind_src].T,
                                     dst[:m, MNN_ind_dst].T)  ## compute the transformation matrix based on MNNs
        import math
        distances = np.mean([math.sqrt(((p1[kk, 0] - p2[kk, 0]) ** 2) + ((p1[kk, 1] - p2[kk, 1]) ** 2))
                             for kk in range(len(p1))])

        # update the current source
        src = np.dot(T, src)
        coor_used = np.dot(T, coor_used)
        coor_all = np.dot(T, coor_all)

        # check error
        mean_error = np.mean(distances)
        # print(mean_error)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    aligned_points = coor_used.T  # MNNs in the landmark_domain
    aligned_points_all = coor_all.T  # all points in the slice

    return key_points_dst, key_points_src, aligned_points[:, :2], aligned_points_all[:, :2]


def ICP_align(adata_concat, adata_target, adata_ref, slice_target, slice_ref, landmark_domain, plot_align=False):
    ### find MNN pairs in the landmark domain with knn=1
    adata_slice1 = adata_target[adata_target.obs['louvain'].isin(landmark_domain)]
    adata_slice2 = adata_ref[adata_ref.obs['louvain'].isin(landmark_domain)]

    batch_pair = adata_concat[
        adata_concat.obs['batch_name'].isin([slice_target, slice_ref]) & adata_concat.obs['louvain'].isin(
            landmark_domain)]
    mnn_dict = create_dictionary_mnn(batch_pair, use_rep='STAligner', batch_name='batch_name', k=1, iter_comb=None,
                                     verbose=0)
    adata_1 = batch_pair[batch_pair.obs['batch_name'] == slice_target]
    adata_2 = batch_pair[batch_pair.obs['batch_name'] == slice_ref]

    anchor_list = []
    positive_list = []
    for batch_pair_name in mnn_dict.keys():
        for anchor in mnn_dict[batch_pair_name].keys():
            positive_spot = mnn_dict[batch_pair_name][anchor][0]
            ### anchor should only in the ref slice, pos only in the target slice
            if anchor in adata_1.obs_names and positive_spot in adata_2.obs_names:
                anchor_list.append(anchor)
                positive_list.append(positive_spot)

    batch_as_dict = dict(zip(list(adata_concat.obs_names), range(0, adata_concat.shape[0])))
    anchor_ind = list(map(lambda _: batch_as_dict[_], anchor_list))
    positive_ind = list(map(lambda _: batch_as_dict[_], positive_list))
    anchor_arr = adata_concat.obsm['STAligner'][anchor_ind,]
    positive_arr = adata_concat.obsm['STAligner'][positive_ind,]
    dist_list = [np.sqrt(np.sum(np.square(anchor_arr[ii, :] - positive_arr[ii, :]))) for ii in
                 range(anchor_arr.shape[0])]

    key_points_src = np.array(anchor_list)[dist_list < np.percentile(dist_list, 50)]  ## remove remote outliers
    key_points_dst = np.array(positive_list)[dist_list < np.percentile(dist_list, 50)]
    # print(len(anchor_list), len(key_points_src))

    coor_src = adata_slice1.obsm["spatial"]  ## to_be_aligned
    coor_dst = adata_slice2.obsm["spatial"]  ## reference_points

    ## index number
    MNN_ind_src = [list(adata_1.obs_names).index(key_points_src[ii]) for ii in range(len(key_points_src))]
    MNN_ind_dst = [list(adata_2.obs_names).index(key_points_dst[ii]) for ii in range(len(key_points_dst))]

    ####### ICP alignment
    init_pose = None
    max_iterations = 100
    tolerance = 0.001

    coor_used = coor_src  ## Batch_list[1][Batch_list[1].obs['annotation']==2].obsm["spatial"]
    coor_all = adata_target.obsm["spatial"].copy()
    coor_used = np.concatenate([coor_used, np.expand_dims(np.ones(coor_used.shape[0]), axis=1)], axis=1).T
    coor_all = np.concatenate([coor_all, np.expand_dims(np.ones(coor_all.shape[0]), axis=1)], axis=1).T
    A = coor_src  ## to_be_aligned
    B = coor_dst  ## reference_points

    m = A.shape[1]  # get number of dimensions

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)
    prev_error = 0

    for ii in range(max_iterations + 1):
        p1 = src[:m, MNN_ind_src].T
        p2 = dst[:m, MNN_ind_dst].T
        T, _, _ = best_fit_transform(src[:m, MNN_ind_src].T,
                                     dst[:m, MNN_ind_dst].T)  ## compute the transformation matrix based on MNNs
        import math
        distances = np.mean([math.sqrt(((p1[kk, 0] - p2[kk, 0]) ** 2) + ((p1[kk, 1] - p2[kk, 1]) ** 2))
                             for kk in range(len(p1))])

        # update the current source
        src = np.dot(T, src)
        coor_used = np.dot(T, coor_used)
        coor_all = np.dot(T, coor_all)

        # check error
        mean_error = np.mean(distances)
        # print(mean_error)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    aligned_points = coor_used.T  # MNNs in the landmark_domain
    aligned_points_all = coor_all.T  # all points in the slice

    if plot_align:
        import matplotlib.pyplot as plt
        plt.rcParams["figure.figsize"] = (3, 3)
        fig, ax = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={'wspace': 0.5, 'hspace': 0.1})
        ax[0].scatter(adata_slice2.obsm["spatial"][:, 0], adata_slice2.obsm["spatial"][:, 1],
                      c="blue", cmap=plt.cm.binary_r, s=1)
        ax[0].set_title('Reference ' + slice_ref, size=14)
        ax[1].scatter(aligned_points[:, 0], aligned_points[:, 1],
                      c="blue", cmap=plt.cm.binary_r, s=1)
        ax[1].set_title('Target ' + slice_target, size=14)

        plt.axis("equal")
        # plt.axis("off")
        plt.show()

    # adata_target.obsm["spatial"] = aligned_points_all[:,:2]
    return aligned_points_all[:, :2]
