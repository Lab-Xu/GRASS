from .color import *
import matplotlib.pyplot as plt
import random
import scanpy as sc


def get_color(n=1, cmap: str = 'scanpy', seed: int = 0):
    r"""
    Get color

    Parameters
    ---------
    n
        number of colors you want
    cmap
        color map (use same with scanpy)
    seed
        random seed to duplicate
    """
    if cmap == 'scanpy' and n <= 10:
        step = 10 // n
        return vega_10_scanpy[::step][:n]
    elif cmap == 'scanpy' and n <= 20:
        step = 20 // n
        return vega_20_scanpy[::step][:n]
    elif cmap == 'scanpy' and n <= 28:
        step = 28 // n
        return zeileis_28[::step][:n]
    elif cmap == 'scanpy' and n <= 102:
        step = 102 // n
        return godsnot_102[::step][:n]
    else:
        print('WARNING: Using random color')
        random.seed(seed)
        if n == 1:
            return "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        elif n > 1:
            return ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n)]


class spatial_domain_visual():
    def __init__(self, adata,
                 slice_name_list, st_list,
                 color_num=10,
                 seed=2024):
        self.adata = adata
        self.slice_name_list = slice_name_list
        self.st_list = st_list
        self.seed = seed
        self.sample_color = get_color(color_num)

    def umap_plot(self, image_path,
                  accuracy_list,
                  accuracy_list_refine,
                  show_list=['slice_name', 'ground_truth']):
        sc.pp.neighbors(self.adata, use_rep='emb', random_state=self.seed)
        sc.tl.umap(self.adata, random_state=self.seed)
        sample_color_dict = dict(zip(self.slice_name_list, self.sample_color))
        self.adata.uns['sample_name_colors'] = [sample_color_dict[x] for x in self.adata.obs.slice_name.cat.categories]

        ari = accuracy_list['all']['ARI']
        ari_refine = accuracy_list_refine['all']['ARI']

        title_list = show_list[:2] + [show_list[2] + ' ari={:.4f}'.format(ari)] + \
                     [show_list[3] + ' ari={:.4f}'.format(ari_refine)]
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = "Arial"
        plt.rcParams["figure.figsize"] = (3, 3)
        plt.rcParams['font.size'] = 12
        sc.pl.umap(self.adata, color=show_list,
                   ncols=4, wspace=0.7,
                   title=title_list,
                   save=image_path, show=False, )

    def domain_identification(self, image_path_part, accuracy_list, accuracy_list_refine,
                              key_list=['louvain']):
        show_list = ['ground_truth'] + key_list
        for index in range(len(self.slice_name_list)):

            slice_name = self.slice_name_list[index]
            image_path = image_path_part + '_{}.png'.format(slice_name)
            temp_ad = self.adata[self.adata.obs['slice_name'] == slice_name]

            for key in key_list:
                if key not in self.st_list[index].obs.keys():
                    self.st_list[index].obs[key] = temp_ad.obs[key]

            ari = accuracy_list[slice_name]['ARI']
            ari_refine = accuracy_list_refine[slice_name]['ARI']
            ari_list = [ari, ari_refine]
            title_list = [f'ground_truth {slice_name}'] + [key_list[i] + ' ari={:.4f}'.format(ari_list[i]) for i in range(len(key_list))]
            sc.pl.spatial(self.st_list[index], color=show_list,
                          ncols=3, wspace=0.7,
                          title=title_list,
                          save=image_path, show=False, )