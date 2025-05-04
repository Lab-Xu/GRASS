from GRASS_ST.emb_clustering import Emb_Clustering


def get_all_emb(adata, 
                nei_index, feats, 
                mps, pos,
                nei_num):

    emb_clustering = Emb_Clustering(nei_index, feats, mps, pos)
    emb_clustering.run_algorithm(sample_rate=[6]*nei_num, nei_num=nei_num)
    emb_clustering.get_embedding(adata)