from sklearn import metrics

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