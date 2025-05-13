import numpy as np
from scipy.sparse import coo_matrix, block_diag
import anndata as ad
import scipy.sparse as sp
import random
import copy
import ot
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn.functional as F

def square_euclid_distance(Z, center, temperature=100):
    ZZ = (Z * Z).sum(-1).reshape(-1, 1).repeat(1, center.shape[0])
    CC = (center * center).sum(-1).reshape(1, -1).repeat(Z.shape[0], 1)
    ZZ_CC = ZZ + CC
    ZC = Z @ center.T
    distance = ZZ_CC - 2 * ZC
    # print("distance:", distance)
    # print("distance shape:", distance.shape)
    return distance / temperature

def high_confidence(Z, center, tao=0.5):
    # print("Z:", Z)
    # print("center:", center)
    distance_norm = torch.min(F.softmax(square_euclid_distance(Z, center), dim=1), dim=1).values
    value, _ = torch.topk(distance_norm, int(Z.shape[0] * (1 - tao)))
    H = torch.where(distance_norm <= value[-1],
                        torch.ones_like(distance_norm), 
                        torch.zeros_like(distance_norm))
    # print("distance_norm  len:", len(distance_norm))
    # print("value len:", len(value))
    # print("distance_norm :", distance_norm)
    # print("value[-1]:", value[-1])
    # print("H sum:", H.sum())
    # print("index shape:", index.shape)
    # print("index:", index)

    # H = torch.nonzero(index).reshape(-1, )
    # print("H shape:", H.shape)
    # print("H:", H)

    return H.cpu()


def low_confidence(Z, center, tao=0.5):

    distance_norm = torch.min(F.softmax(square_euclid_distance(Z, center), dim=1), dim=1).values
    value, _ = torch.topk(distance_norm, int(Z.shape[0] * (1 - tao)))
    L = torch.where(distance_norm > value[-1],
                        torch.ones_like(distance_norm), 
                        torch.zeros_like(distance_norm))

    # L = torch.nonzero(index).reshape(-1, )

    return L.cpu()

def pairwise_distance(data1, data2, device=torch.device('cuda')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    # print("dis inner:", dis.shape)
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cuda')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis


def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state

def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        device=torch.device('cuda:0')
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    # print(f'running k-means on {device}..')
    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # print('X shape:', X.shape)

    # initialize
    dis_min = float('inf')
    initial_state_best = None
    for i in range(20):
        initial_state = initialize(X, num_clusters)
        dis = pairwise_distance_function(X, initial_state).sum()
        if dis < dis_min:
            dis_min = dis
            initial_state_best = initial_state

    initial_state = initial_state_best
    iteration = 0
    while True:
        dis = pairwise_distance_function(X, initial_state)
        # print("dis dim: ", dis.shape)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        if iteration > 500:
            break
        if center_shift ** 2 < tol:
            break

    return choice_cluster.cpu(), initial_state


def calcu_adj(cord, cord2=None, n_neigh=10, metric='minkowski'):
    '''
    Construct adjacency matrix with coordinates.
    input: cord, np.array
    '''
    output_graph = False

    if cord2 is None:
        cord2 = cord
        output_graph = True
        n_neigh += 1

    neigh = NearestNeighbors(n_neighbors=n_neigh, metric=metric).fit(cord2)
    neigh_index = neigh.kneighbors(cord, return_distance=False)
    if output_graph:
        neigh_graph = neigh.kneighbors_graph(cord).toarray()
    else:
        neigh_graph = None

    return neigh_index, neigh_graph

def get_data_list(adata, st_name_list, 
                  n_neigh=10,
                  latent_key='X_batch_removal', 
                  input_data_key='counts',
                  spatial_key='spatial',
                  batch_key='slice_name',
                  cluster_num=None,
                  device=None,):

    nei_index_dict = {}
    nei_graph_dict = {}
    feats_dict = {}

    for name_source in st_name_list:
        adata_source = adata[adata.obs[batch_key] == name_source]
        feat = adata_source.obsm[input_data_key]
        feats_dict[name_source] = feat

        for name_target in st_name_list:
            adata_target = adata[adata.obs[batch_key] == name_target]

            feat_source = adata_source.obsm[latent_key]
            feat_target = adata_target.obsm[latent_key]
            cord_source = adata_source.obsm[spatial_key]
            cord_target = adata_target.obsm[spatial_key]
            
            if name_source != name_target:
                nei_index_dict[name_source, '1', name_target], _ = calcu_adj(cord=feat_source, cord2=feat_target, n_neigh=n_neigh)
            else:
                nei_index_dict[name_source, '0', name_target], nei_graph_dict[name_source] = calcu_adj(cord=cord_source, n_neigh=n_neigh)

    return nei_index_dict, nei_graph_dict, feats_dict

def get_adj(X_data,
            coor_data=None,
            k_X=0,
            k_C=10,
            weight=0,
            include_self_X=False,
            include_self_C=False,
            ):
    from sklearn.neighbors import kneighbors_graph
    A_X = kneighbors_graph(X_data, n_neighbors=k_X, include_self=include_self_X)
    A_C = kneighbors_graph(coor_data, n_neighbors=k_C, include_self=include_self_C)
    graph_neigh = weight * A_X + (1 - weight) * A_C
    graph_neigh = graph_neigh.toarray()
    A = (graph_neigh + graph_neigh.T) / 2

    return A, graph_neigh

def construct_interaction(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']

    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]

    adata.obsm['distance_matrix'] = distance_matrix

    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1

    adata.obsm['graph_neigh'] = interaction

    # transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    adata.obsm['adj'] = adj


def construct_interaction_KNN(adata, n_neighbors=3):
    position = adata.obsm['spatial']
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(position)
    _, indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1

    adata.obsm['graph_neigh'] = interaction

    # transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    adata.obsm['adj'] = adj
    print('Graph constructed!')


def aug_graph(X, A, aug_type=None, drop_percent=0.1):
    # print('Augmenting Feature or Adj')
    if aug_type == 'edge':
        aug_A = aug_random_edge(A, drop_percent=drop_percent)
        aug_X = X
    elif aug_type == 'mask':
        aug_X = aug_random_mask(X, drop_percent=drop_percent)
        aug_A = A
    return aug_X, aug_A


def aug_random_edge(input_adj, drop_percent=0.1):
    percent = drop_percent / 2
    row_idx, col_idx = input_adj.nonzero()

    index_list = [(row_idx[i], col_idx[i]) for i in range(len(row_idx))]

    single_index_list = []
    for i in list(index_list):
        single_index_list.append(i)
        index_list.remove((i[1], i[0]))

    edge_num = int(len(row_idx) / 2)  # 9228 / 2
    add_drop_num = int(edge_num * percent / 2)
    print("droped or added edge number:", add_drop_num)

    aug_adj = copy.deepcopy(input_adj.todense().tolist())

    edge_idx = [i for i in range(edge_num)]

    drop_idx = random.sample(edge_idx, add_drop_num)

    for i in drop_idx:
        aug_adj[single_index_list[i][0]][single_index_list[i][1]] = 0
        aug_adj[single_index_list[i][1]][single_index_list[i][0]] = 0

    '''
    above finish drop edges
    '''
    node_num = input_adj.shape[0]
    l = [(i, j) for i in range(node_num) for j in range(i)]
    add_list = random.sample(l, add_drop_num)

    for i in add_list:
        aug_adj[i[0]][i[1]] = 1
        aug_adj[i[1]][i[0]] = 1

    aug_adj = np.matrix(aug_adj)
    aug_adj = coo_matrix(aug_adj)
    return aug_adj


def aug_random_mask(input_feature, drop_percent=0.1):
    node_num = input_feature.shape[0]
    mask_num = int(node_num * drop_percent)
    print("masked row number:", mask_num)

    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    aug_feature = copy.deepcopy(input_feature)
    zeros = np.zeros_like(aug_feature[0])
    for j in mask_idx:
        aug_feature[j] = zeros
    return aug_feature
