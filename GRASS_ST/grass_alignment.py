import copy
import numpy as np
from scipy.sparse import coo_matrix, block_diag
import torch
from sklearn.neighbors import NearestNeighbors
import ot
import networkx as nx
import pandas as pd
import networkx as nx
from typing import Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
from scipy.spatial import cKDTree
from GRASS_ST.progressive_alignment import GradAlign
from GRASS_ST.reconstruction_tissue import ICP_3D_reconstruct
import logging

logging.basicConfig(level=logging.INFO)



def evaluate_align_performance(adata_list, slice_name_list, matching_list, evaluate_type='same_cluster'):
    accuracy_list = []
    if evaluate_type == 'same_cluster':
        print("evaluate alignment | ")
        for index in range(len(matching_list)):
            corr_num = 0
            source_adata = adata_list[index]
            target_adata = adata_list[index + 1]
            source_name = slice_name_list[index]
            target_name = slice_name_list[index + 1]
            matching = matching_list[source_name]
            for k, v in matching.items():
                if source_adata.obs['ground_truth'][k] == target_adata.obs['ground_truth'][v]:
                    corr_num += 1
            accuracy = corr_num / len(matching)
            print("alignment {}-{}:{:.4f}".format(source_name, target_name, accuracy))
            accuracy_list.append(accuracy)
        print("average accuracy:{:.4f}".format(np.mean(accuracy_list)))

    return accuracy_list


def create_node_index_maps(G1, G2):

    node_list1 = list(G1.nodes())
    node_list2 = list(G2.nodes())

    idx1_dict = {node: idx for idx, node in enumerate(node_list1)}
    idx2_dict = {node: idx for idx, node in enumerate(node_list2)}

    return idx1_dict, idx2_dict

def shift_node_indices(graph: nx.Graph, shift_value: int):
    """
    Shift node indices of a graph by a given value.
    """
    shifted_nodes = {node: node + shift_value for node in graph.nodes()}
    return nx.relabel_nodes(graph, shifted_nodes)

def update_alignment_dict(alignment_dict: Dict[int, int], shift_value: int):
    """
    Update the alignment dictionary with new node indices after shifting.
    """
    shifted_alignment_dict = {source_node: target_node + shift_value for source_node, target_node in alignment_dict.items()}
    reversed_alignment_dict = {value: key for (key, value) in shifted_alignment_dict.items()}
    return shifted_alignment_dict, reversed_alignment_dict

def preprocess_graphs(source_graph: nx.Graph, target_graph: nx.Graph, alignment_dict: Dict[int, int]):
    """
    Preprocess the source and target graphs and their alignment dictionary.
    """
    shift_value = source_graph.number_of_nodes()
    shifted_target_graph = shift_node_indices(target_graph, shift_value)
    
    updated_alignment_dict, reversed_alignment_dict = update_alignment_dict(alignment_dict, shift_value)
    
    return shifted_target_graph, updated_alignment_dict, reversed_alignment_dict


def load_attribute(feature1, feature2, G1, G2):
    
    G1_nodes = list(G1.nodes())
    G2_nodes = list(G2.nodes())

    feature1_df = pd.DataFrame(feature1)
    feature2_df = pd.DataFrame(feature2)

    attribute1 = np.array(feature1_df.loc[G1_nodes, :])
    attribute2 = np.array(feature2_df.loc[G2_nodes, :])

    attr_cos = cosine_similarity(attribute1, attribute2)

    return attr_cos, attribute1, attribute2

def read_attribute(feature1, feature2, G1, G2):
    try:
        attribute, attr1, attr2 = load_attribute(feature1, feature2, G1, G2)
        # attribute, attr1, attr2, attr1_pd, attr2_pd = load_attribute_for_perturb(attribute_folder, filename, G1, G2,alignment_dict, 0.2)
        attribute = attribute.transpose()
    except:
        attr1 = []
        attr2 = []
        attribute = []
        print('Attribute files not found.')
    return attr1, attr2, attribute

def build_alignment_dictionaries(alignment_df):
    forward_alignment_dict = {row[0]: row[1] for row in alignment_df.itertuples(index=False)}
    reverse_alignment_dict = {row[1]: row[0] for row in alignment_df.itertuples(index=False)}
    return forward_alignment_dict, reverse_alignment_dict


def pre_alignment(matched_nodes):
    # alignment = {matched_nodes[k]: k for k in matched_nodes.keys()}

    data = {
        '0': list(matched_nodes.keys()),
        '1': list(matched_nodes.values()),
    }
    alignment_df = pd.DataFrame(data)
    return alignment_df

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

def get_adj(coor_data,
            k=10,
            include_self_C=False,
            ):

    graph_neigh = kneighbors_graph(coor_data, n_neighbors=k, include_self=include_self_C)
    graph_neigh = graph_neigh.toarray()
    A = (graph_neigh + graph_neigh.T) / 2

    return A, graph_neigh

def get_graph(st_adata_list, slice_name_list, method_type='GraphST', platform='10X', k=10):

    slice_name = slice_name_list[0]

    # print(f'=====Construct multiple graphs for slice {slice_name}=====')

    adj_list = []

    for slice, st_adata in zip(slice_name_list, st_adata_list):

        if slice==slice_name:
            features = st_adata.X
            use_st_adata = st_adata.copy()

        if method_type=='original':
            Adj, _ = get_adj(coor_data=st_adata.obsm["spatial"], k=k)
            adj_list.append(coo_matrix(Adj, dtype='float32'))

        elif method_type=='GraphST':
            if platform in ['Stereo', 'Slide']:
                construct_interaction_KNN(st_adata)
            else:
                construct_interaction(st_adata)
            adj_list.append(coo_matrix(st_adata.obsm['adj']))

    return adj_list, features, use_st_adata

def find_mutual_nn(source, target, num_neighbor=20):

    tree_target = cKDTree(target)
    _, indices_source = tree_target.query(source, k=num_neighbor)
    # print("indices_source shape:", indices_source.shape)

    tree_source = cKDTree(source)
    _, indices_target = tree_source.query(target, k=num_neighbor)
    # print("indices_target shape:", indices_target.shape)

    match1 = {(a, b_i) for a, b in enumerate(indices_source) for b_i in b}
    match2 = {(a, b_i) for a, b in enumerate(indices_target) for b_i in b}
    mutual = match1 & set([(b, a) for a, b in match2])

    return mutual


def get_align_point(source_data, target_data, loc_name=None, emb_name=None,
                    num_neighbor_loc=20, num_neighbor_emb=20,
                    loc_use=True, emb_use=True):
    if loc_use & emb_use:
        loc_source = source_data.obsm[loc_name]
        loc_target = target_data.obsm[loc_name]
        mutual_loc = find_mutual_nn(loc_source, loc_target, num_neighbor=num_neighbor_loc)

        emb_source = source_data.obsm[emb_name]
        emb_target = target_data.obsm[emb_name]
        mutual_emb = find_mutual_nn(emb_source, emb_target, num_neighbor=num_neighbor_emb)

        align_point = mutual_loc & mutual_emb

        return align_point

def get_anchor(align_point):
    source_point_index = []
    target_point_index = []
    align_key_point = {}
    for i, j in align_point:
        if i in source_point_index:
            continue
        else:
            source_point_index.append(i)
        if j in target_point_index:
            continue
        else:
            target_point_index.append(j)
        align_key_point[i] = j
    return align_key_point

class GRASS_Alignment(object):
    def __init__(
        self,
        adata_original,
        slice_name_list,
        st_ad_list,
        cluster_method='mclust',
        k_hop=2, 
        beta=0.01, 
        fast_mode=True,
        seed_add_step=10,
        align_iter_num=10,
        seed=2024,
    ):
        self.st_list = st_ad_list.copy()
        self.adata = adata_original.copy()
        self.slice_name_list = slice_name_list
        self.cluster_method = cluster_method
        self.source_anchor_dict = {}
        self.align_graph_dict = {}
        self.source_matching_dict = {}

        # Parameters for alignment
        self.k_hop = k_hop
        self.beta = beta
        self.fast_mode = fast_mode
        self.align_iter_num = align_iter_num
        self.seed_add_step = seed_add_step

        # print(f"st_list[0] {cluster_method}：", self.st_list[0].obs[cluster_method])

        for index in range(len(slice_name_list)):
            # embedding
            temp_ad = self.adata[self.adata.obs['slice_name'] == slice_name_list[index]]
            self.st_list[index].obs[cluster_method] = temp_ad.obs[cluster_method]

            self.st_list[index].obsm['embeds'] = temp_ad.obsm['embeds']

            raw_loc = self.st_list[index].obsm['spatial']
            raw_loc[:, :2] = raw_loc[:, :2] - np.median(raw_loc[:, :2], axis=0)
            self.st_list[index].obsm['spatial_pair'] = raw_loc
        
        self.adata.obs_names = [x + '_' + self.adata.obs['slice_name'][i] for i, x in enumerate(self.adata.obs_names)]

        # print(f'adata list: \n{self.st_list}')
        # print(f"st_list[0] {cluster_method}：", self.st_list[0].obs[cluster_method])
    
    def get_anchors(self,
                    num_neighbor_loc=50,
                    num_neighbor_emb=50,
                    same_cluster=False):
        
        if len(self.slice_name_list) < 2:
            logging.warning("Not enough slices to find anchors.")
            return None

        for i in range(1, len(self.slice_name_list)):
            source_data = self.st_list[i - 1]
            target_data = self.st_list[i]
            align_point = get_align_point(source_data, target_data, loc_name='spatial_pair', emb_name='embeds',
                                          num_neighbor_loc=num_neighbor_loc,
                                          num_neighbor_emb=num_neighbor_emb,
                                          )
            anchor = get_anchor(align_point)

            if len(anchor) == 0:
                align_point = get_align_point(source_data, target_data, loc_name='spatial_pair', emb_name='embeds',
                                              num_neighbor_emb=num_neighbor_emb,
                                              loc_use=False)
                anchor = get_anchor(align_point)
            
            logging.info(f'source: {self.slice_name_list[i - 1]}, target: {self.slice_name_list[i]} | anchor number: {len(anchor.keys())}')

            if same_cluster:
                filtered_points = []
                removed_count = 0
                align_point_item = copy.deepcopy(align_point)
                for item in align_point_item:
                    s = source_data.obs[self.cluster_method][item[0]]
                    t = target_data.obs[self.cluster_method][item[1]]
                    if s != t:
                        align_point.remove(item)
                        removed_count += 1
                    else:
                        filtered_points.append(item)
                logging.info(f"{removed_count} items were removed.")
                logging.info(f"{len(align_point)} items were reserved.")
                anchor = get_anchor(align_point)

            self.source_anchor_dict[self.slice_name_list[i - 1]] = anchor
            # print('source:{}, target:{}|anchor number:{}'.format(self.slice_name_list[i - 1],
            #                                                      self.slice_name_list[i],
            #                                                      len(anchor)))
        # return anchor

    def construct_multi_graph(self):
        num_slices = len(self.slice_name_list)

        for index in range(num_slices):
            st_adata_split_list = []
            use_slice_name_list = []
            slice_name = self.slice_name_list[index]
            source_adata = self.st_list[index]

            if index == num_slices - 1:
                slice_name_anterior = self.slice_name_list[index - 1]
                adata_anterior = self.st_list[index - 1]

                anterior_anchor = self.source_anchor_dict[slice_name_anterior]
                anterior_index = list(anterior_anchor.keys())
                source_index = list(anterior_anchor.values())

                anterior_adata_split = adata_anterior[anterior_index].copy()
                source_adata_split = source_adata[source_index].copy()

                st_adata_split_list.extend([source_adata_split, anterior_adata_split])
                use_slice_name_list.extend([slice_name, slice_name_anterior])

            elif index==0:
                slice_name_posterior = self.slice_name_list[index + 1]
                adata_posterior = self.st_list[index + 1]

                source_anchor = self.source_anchor_dict[slice_name]

                source_index = list(source_anchor.keys())
                posterior_index = list(source_anchor.values())

                source_adata_split = source_adata[source_index].copy()
                posterior_adata_split = adata_posterior[posterior_index].copy()

                st_adata_split_list.extend([source_adata_split, posterior_adata_split])
                use_slice_name_list.extend([slice_name, slice_name_posterior])

            else:
                slice_name_anterior = self.slice_name_list[index - 1]
                adata_anterior = self.st_list[index - 1]

                slice_name_posterior = self.slice_name_list[index + 1]
                adata_posterior = self.st_list[index + 1]

                anterior_anchor = self.source_anchor_dict[slice_name_anterior]
                source_anchor = self.source_anchor_dict[slice_name]

                common_values = set(anterior_anchor.values()) & set(source_anchor.keys())
                anterior_anchor_filter = {k: v for k, v in anterior_anchor.items() if v in common_values}
                source_anchor_filter = {k: v for k, v in source_anchor.items() if k in common_values}

                anterior_index = list(anterior_anchor_filter.keys())
                source_index = list(anterior_anchor_filter.values())
                posterior_index = list(source_anchor_filter.values())

                anterior_adata_split = adata_anterior[anterior_index].copy()
                source_adata_split = source_adata[source_index].copy()
                posterior_adata_split = adata_posterior[posterior_index].copy()

                st_adata_split_list.extend([source_adata_split, 
                                            anterior_adata_split, 
                                            posterior_adata_split])

                use_slice_name_list.extend([slice_name, 
                                            slice_name_anterior,
                                            slice_name_posterior])

            # print(f"slice {slice_name} havs {len(source_index)} nodes")

            adj_list, features, use_st_adata = get_graph(st_adata_split_list, use_slice_name_list)
            self.align_graph_dict[slice_name] = (adj_list, features, use_st_adata)


    def _build_graph(self, features, adj_matrix):
        """
        Generate graph
        """
        G = nx.Graph()
        for index, feature in enumerate(features):
            G.add_node(index, feature=feature)
        
        # Check if adj_matrix is a sparse tensor
        if adj_matrix.is_sparse:
            # Convert sparse tensor to dense tensor
            adj_matrix = adj_matrix.to_dense().cpu().numpy()

        edges = np.transpose(np.nonzero(adj_matrix))
        edges = [tuple(edge) for edge in edges]

        # print('edges:', edges)
        G.add_edges_from(edges)
        return G

    def align_multi_slice(self, adj_list, method_type='grad_align'):

        if method_type == 'grad_align':

            for i in range(1, len(self.slice_name_list)):
                source_slice_name, target_slice_name = self.slice_name_list[i - 1], self.slice_name_list[i]
                print("==========alignment spot=============")
                print("source slice:{}, target slice:{}".format(source_slice_name, target_slice_name))
                source_data, target_data = self.st_list[i - 1], self.st_list[i]
                feature_source = source_data.obsm['embeds']
                feature_target = target_data.obsm['embeds']

                G_source, G_target = self._build_graph(feature_source, adj_list[i - 1]), \
                                                    self._build_graph(feature_target, adj_list[i])
                print(f"G_source:{G_source}, G_target:{G_target}")

                # alignment
                idx1_dict, idx2_dict, seed_list1, seed_list2 = self.grad_align_slice(source_slice_name,
                                                                                     G_source,
                                                                                     G_target,
                                                                                     feature_source,
                                                                                     feature_target)

                # print("type seed_list1:{}, number seed_list1:{}".format(type(seed_list1), len(seed_list1)))
                # print("type seed_list2:{}, number seed_list2:{}".format(type(seed_list1), len(seed_list1)))

                seed_list1_sort = [idx1_dict[item] for item in seed_list1]
                seed_list2_sort = [idx2_dict[item] for item in seed_list2]

                matching = dict([(k, v) for k, v in zip(seed_list1_sort, seed_list2_sort)])

                self.source_matching_dict[source_slice_name] = matching

    def grad_align_slice(self, source_slice_name,
                         G_source, G_target,
                         feature_source, feature_target):
        # fast_mode = self.fast_mode if len(self.source_anchor_dict[source_slice_name]) > 0 else False
        # print("feature_source:", feature_source)
        print("feature_source shape:", feature_source.shape)

        alignment_df = pre_alignment(self.source_anchor_dict[source_slice_name])
        alignment_dict, alignment_dict_reversed = build_alignment_dictionaries(alignment_df)
        attr1, attr2, _ = read_attribute(feature_source, feature_target, G_source, G_target)
        G_target, alignment_dict, alignment_dict_reversed = preprocess_graphs(G_source, G_target, alignment_dict)
        idx1_dict, idx2_dict = create_node_index_maps(G_source, G_target)
        emb1 = torch.tensor(feature_source, dtype=torch.float32)
        emb2 = torch.tensor(feature_target, dtype=torch.float32)

        align_spot = GradAlign(G_source, G_target, attr1, attr2, 
                                alignment_dict, alignment_dict_reversed,
                                idx1_dict, idx2_dict,
                                k_hop=self.k_hop,
                                alpha=G_target.number_of_nodes() / G_source.number_of_nodes(),
                                beta=self.beta, 
                                fast_mode=self.fast_mode,
                                seed_add_step=self.seed_add_step,
                                align_iteration=self.align_iter_num)

        _, _, seed_list1, seed_list2, _ = align_spot.run_algorithm(emb1, emb2)

        return idx1_dict, idx2_dict, seed_list1, seed_list2

    def reconstruct_3D_tissue(self, cluster_align_list, 
                              method_type='ICP',
                              cluster_method='mclust'):
        key_points = {}
        aligned_points_dict = {}
        aligned_points_all_dict = {}

        if method_type == 'ICP':

            for i in range(1, len(self.slice_name_list)):
                source_slice_name = self.slice_name_list[i - 1]
                target_slice_name = self.slice_name_list[i]
                source_cluster = cluster_align_list[i - 1]
                target_cluster = cluster_align_list[i]
                print("==========3D reconstruction==========")
                print("source slice:{}, target slice:{}".format(source_slice_name, target_slice_name))
                source_data = self.st_list[i - 1]
                target_data = self.st_list[i]
                matching_dict = self.source_matching_dict[source_slice_name]
                key_points_src, key_points_dst, aligned_points, aligned_points_all = ICP_3D_reconstruct(
                    source_data, target_data, source_cluster, target_cluster, matching_dict,
                    cluster_method=cluster_method)

                aligned_points_dict[target_slice_name] = aligned_points
                aligned_points_all_dict[target_slice_name] = aligned_points_all
                key_points[source_slice_name] = dict(zip(key_points_src, key_points_dst))
        else:
            logging.error("Unsupported method type: {}".format(method_type))
            raise ValueError("Unsupported method type. Only 'ICP' is currently supported.")


        return key_points, aligned_points_dict, aligned_points_all_dict
