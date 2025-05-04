import numpy as np
import torch
import torch.backends.cudnn as cudnn
import warnings
import datetime
import pickle as pkl
import os
import random
import copy
import scanpy as sc
from sklearn import metrics
import anndata as ad
import pandas as pd
from itertools import chain

from GRASS_ST.emb_clustering import Emb_Clustering
from GRASS_ST.get_emb import get_all_emb
from GRASS_ST.utils import preprocess, construct_graph, cluster_func, batch_effect_removal
import tool.metrics_index as tool_metric


def cluster_performance(ad, annotation_key, cluster_key, slice_name):

    real_y = ad.obs[annotation_key][~ad.obs[annotation_key].isna()]
    pre_y = ad.obs[cluster_key][~ad.obs[annotation_key].isna()]
    # pre_y = [str(label) for label in pre_y]

    ari_score = metrics.adjusted_rand_score(real_y, pre_y)
    nmi_score = metrics.normalized_mutual_info_score(real_y, pre_y)
    ac_score = metrics.accuracy_score(real_y, pre_y)

    # jac_score = metrics.jaccard_score(real_y, pre_y, average='macro')
    print("{} | ARI:{:.4f}, NMI:{:.4f}, AC:{:.4f}".format(slice_name, ari_score, nmi_score, ac_score))
    return ari_score, nmi_score, ac_score

def evaluate_cluster_performance(adata, 
                                 annotation_key='ground_truth', 
                                 cluster_key='mclust', 
                                 slice_key='batch'):
    
    print(f"=====Metrics for {cluster_key}=====")
    accuracy_list = {'all': {}}
    ari_score, nmi_score, ac_score = cluster_performance(adata, 
                                                         annotation_key=annotation_key,
                                                         cluster_key=cluster_key, 
                                                         slice_name='all')
    accuracy_list['all']['ARI'] = ari_score
    accuracy_list['all']['NMI'] = nmi_score
    accuracy_list['all']['AC'] = ac_score

    for slice_name in list(adata.obs[slice_key].unique()):
        temp_ad = adata[adata.obs[slice_key] == slice_name]
        ari_score, nmi_score, ac_score = cluster_performance(temp_ad, 
                                                   annotation_key=annotation_key,
                                                   cluster_key=cluster_key, 
                                                   slice_name=slice_name)
        
        accuracy_list[slice_name] = {}
        accuracy_list[slice_name]['ARI'] = ari_score
        accuracy_list[slice_name]['NMI'] = nmi_score
        accuracy_list[slice_name]['AC'] = ac_score

    return accuracy_list

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def run(st_list, st_name_list,
        have_annotation=False,
        filter_genes=False,
        detect_NaN=False,
        batch_key='slice_name',
        batch_removal_method='harmony',
        n_comps=32,
        epoch_ae=100,
        n_neigh=10,
        spatial_key='spatial',
        latent_key='X_batch_removal',
        weight_graph=False,
        input_data_key='counts',
        same_mp=True,
        same_sc=False,
        # batch_size=128,
        hidden_dim_list=[64],
        nb_epochs=1000,
        patience=20,
        num_samples=None,
        num_samples_inter=None,
        tao=0.9,
        w_pos=1, w_neg=0,
        w_intra=1, w_re=1, w_inter=1,
        w_lg=1,
        cluster_num=None,
        cluster_method='mclust',
        cluster_name='mclust',
        use_remove_batch=False,
        refinement=True,
        n_neighbors_refine=50,
        refine_method='pot',
        refine_name=None,
        device=None,
        seed=2024,
        ):
    
    print("===========param setting===========")
    print(f"batch_removal:{batch_removal_method}, n_neigh:{n_neigh}, \
          hidden_dim_list:{hidden_dim_list}, nb_epochs:{nb_epochs}, \
          ww_intra_c:{w_intra}, w_re:{w_re}, w_inter:{w_inter},  \
            cluster_method:{cluster_method}, seed:{seed}")
    
    nei_num=len(st_name_list)-1
    
    fix_seed(seed)

    adata = preprocess(st_list, st_name_list,
                       have_annotation=have_annotation,
                       filter_genes=filter_genes,
                       detect_NaN=detect_NaN)

    batch_effect_removal(adata, 
                         method_type=batch_removal_method,
                         latent_key=latent_key,
                         batch_key=batch_key,
                         n_comps=n_comps,
                         epoch_ae=epoch_ae,
                         device=device)
    

    print('construct graph...')
    graph_dict = construct_graph(adata, st_name_list,
                                n_neigh=n_neigh,
                                tao=tao,
                                spatial_key=spatial_key,
                                latent_key=latent_key,
                                input_data_key=input_data_key,
                                weight_graph=weight_graph,
                                cluster_num=cluster_num,
                                device=device)
    print(f'========GRASS========')
    emb_clustering = Emb_Clustering(st_name_list, graph_dict,
                                    nb_epochs=nb_epochs,
                                    patience=patience,
                                    device=device)
    print('start train...')
    emb_clustering.run_algorithm(hidden_dim_list=hidden_dim_list,
                                sample_rate=[6]*nei_num,
                                nei_num=nei_num, 
                                w_pos=w_pos, w_neg=w_neg,
                                w_intra=w_intra, w_re=w_re, w_inter=w_inter,
                                same_mp=same_mp, same_sc=same_sc,
                                num_samples=num_samples, num_samples_inter=num_samples_inter)
    print('end train...')
    print("get embedding...")
    emb_clustering.get_embedding(adata, st_list)

    print("identify spatial domain")
    emb_clustering.identify_spatial_domain(adata, 
                                           cluster_num=cluster_num,
                                           cluster_method=cluster_method,
                                           cluster_name=cluster_name,
                                           use_remove_batch=use_remove_batch,
                                           refinement=refinement,
                                           refine_method=refine_method,
                                           refine_name=refine_name,
                                           n_neighbors_refine=n_neighbors_refine)
    if have_annotation:
        metrics_dict = evaluate_cluster_performance(adata,
                                                    annotation_key='ground_truth',
                                                    cluster_key=cluster_method,
                                                    slice_key=batch_key)
        if refinement:
            refine_metrics_dict = evaluate_cluster_performance(adata,
                                                                annotation_key='ground_truth',
                                                                cluster_key=refine_name,
                                                                slice_key=batch_key)
            
    else:
        metrics_dict = None
        refine_metrics_dict = None


    return adata, metrics_dict, refine_metrics_dict, emb_clustering.loss_list

