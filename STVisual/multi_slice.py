r"""
Vis multi dataset and their connection
"""
import random
import math
from typing import List, Mapping, Optional, Union
import scanpy as sc
import numpy as np
import pandas as pd
from anndata import AnnData

import matplotlib
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

from .color import *


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


class build_3D():
    r"""
    Build 3D pics/models from multi-datasets

    Parameters
    ---------
    datasets
        list adata of in order
    mappings
        list of SLAT matching results
    spatial_key
        obsm key of spatial info
    anno_key
        obs key of cell annotation such as celltype
    subsample_size
        subsample size of matches
    scale_coordinate
        scale the coordinate from different slides
    """

    def __init__(self, adatas: List[AnnData],
                 mappings: List[np.ndarray],
                 spatial_key: Optional[str] = 'spatial',
                 anno_key: Optional[str] = 'annotation',
                 subsample_size: Optional[int] = 200,
                 scale_coordinate: Optional[bool] = True,
                 ) -> None:
        assert len(mappings) == len(adatas) - 1
        adatas = adatas.copy()

        self.mappings = mappings
        self.loc_list = []
        self.anno_list = []
        for adata in adatas:
            loc = adata.obsm[spatial_key].copy()
            loc = loc.astype('float')
            if scale_coordinate:
                for i in range(2):
                    loc[:, i] = (loc[:, i] - np.min(loc[:, i])) / (np.max(loc[:, i]) - np.min(loc[:, i]))
            anno = adata.obs[anno_key]
            self.loc_list.append(loc)
            self.anno_list.append(anno)

        self.celltypes = set(pd.concat(self.anno_list))
        self.subsample_size = subsample_size

    def draw_3D(self,
                size: Optional[List[int]] = [10, 10],
                point_size: Optional[List[int]] = [0.5, 0.5],
                point_alpha: Optional[float] = 0.6,
                line_width: Optional[float] = 0.6,
                line_color: Optional[str] = '#4169E1',
                line_alpha: Optional[float] = 0.8,
                hide_axis: Optional[bool] = False,
                height: Optional[float] = 1.0,
                height_scale: Optional[float] = 1.0,
                have_color_map=False,
                color_map=None,
                save:Optional[str]=None,
                ) -> None:
        r"""
        Draw 3D picture of two layers

        Parameters:
        ----------
        size
            plt figure size (width, height)
        point_size
            point size of each layer
        point_alpha
            point alpha of each layer
        line_width
            pair line width
        line_color
            pair line color
        line_alpha
            pair line alpha
        hide_axis
            if hide axis
        height
            height of one layer
        """
        fig = plt.figure(figsize=(size[0], size[1]))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, height_scale * len(self.mappings)])
        
        if have_color_map:
            c_map = color_map
        else:
            # color by different cell types
            color = get_color(len(self.celltypes))
            c_map = {}
            for i, celltype in enumerate(self.celltypes):
                c_map[celltype] = color[i]

        for j, mapping in enumerate(self.mappings):
            print(f"Mapping {j}th layer ")
            # plot cells
            for i, (layer, anno) in enumerate(zip(self.loc_list[j:j + 2], self.anno_list[j:j + 2])):
                if i == 0 and 0 < j < len(self.mappings) - 1:
                    continue
                for cell_type in self.celltypes:
                    slice = layer[anno == cell_type, :]
                    xs = slice[:, 0]
                    ys = slice[:, 1]
                    zs = height * (j + i)
                    ax.scatter(xs, ys, zs, s=point_size[i], c=c_map[cell_type], alpha=point_alpha)
            # plot mapping line
            mapping = mapping[:, np.random.choice(mapping.shape[1], self.subsample_size, replace=False)].copy()
            for k in range(mapping.shape[1]):
                cell1_index = mapping[:, k][0]  # query
                cell0_index = mapping[:, k][1]  # ref
                cell0_coord = self.loc_list[j][cell0_index, :]
                cell1_coord = self.loc_list[j + 1][cell1_index, :]
                coord = np.row_stack((cell0_coord, cell1_coord))
                ax.plot(coord[:, 0], coord[:, 1], [height * j, height * (j + 1)], color=line_color,
                        linestyle="dashed", linewidth=line_width, alpha=line_alpha)

        if hide_axis:
            plt.axis('off')

        if save != None:
            print("save image")
            plt.savefig(save)

        plt.show()


class match_3D_multi():
    r"""
    Plot the mapping result between 2 datasets

    Parameters
    ---------
    dataset_A
        pandas dataframe which contain ['index','x','y'], reference dataset
    dataset_B
        pandas dataframe which contain ['index','x','y'], target dataset
    matching
        matching results
    meta
        dataframe colname of meta, such as celltype
    expr
        dataframe colname of gene expr
    subsample_size
        subsample size of matches
    reliability
        match score (cosine similarity score)
    scale_coordinate
        if scale coordinate via (:math:`data - np.min(data)) / (np.max(data) - np.min(data))`)
    rotate
        how to rotate the slides (force scale_coordinate), such as ['x','y'], means dataset0 rotate on x axes
        and dataset1 rotate on y axes
    change_xy
        exchange x and y on dataset_B
    subset
        index of query cells to be plotted

    Note
    ----------
    dataset_A and dataset_B can in different length

    """

    def __init__(self, dataset_A: pd.DataFrame,
                 dataset_B: pd.DataFrame,
                 matching: np.ndarray,
                 meta: Optional[str] = None,
                 expr: Optional[str] = None,
                 subsample_size: Optional[int] = 300,
                 reliability: Optional[np.ndarray] = None,
                 scale_coordinate: Optional[bool] = True,
                 rotate: Optional[List[str]] = None,
                 exchange_xy: Optional[bool] = False,
                 subset: Optional[List[int]] = None
                 ) -> None:
        self.dataset_A = dataset_A.copy()
        self.dataset_B = dataset_B.copy()
        self.meta = meta
        self.matching = matching
        self.conf = reliability
        self.subset = subset  # index of query cells to be plotted
        scale_coordinate = True if rotate != None else scale_coordinate

        assert all(item in dataset_A.columns.values for item in ['index', 'x', 'y'])
        assert all(item in dataset_B.columns.values for item in ['index', 'x', 'y'])

        if meta:
            set1 = list(set(self.dataset_A[meta]))
            set2 = list(set(self.dataset_B[meta]))
            self.celltypes = set1 + [x for x in set2 if x not in set1]
            self.celltypes.sort()  # make sure celltypes are in the same order
            overlap = [x for x in set2 if x in set1]
            # print(f"dataset1: {len(set1)} cell types; dataset2: {len(set2)} cell types; \n\
            #         Total :{len(self.celltypes)} celltypes; Overlap: {len(overlap)} cell types \n\
            #         Not overlap :[{[y for y in (set1 + set2) if y not in overlap]}]"
            #       )
        self.expr = expr if expr else False

        if scale_coordinate:
            for i, dataset in enumerate([self.dataset_A, self.dataset_B]):
                for axis in ['x', 'y']:
                    dataset[axis] = (dataset[axis] - np.min(dataset[axis])) / (
                            np.max(dataset[axis]) - np.min(dataset[axis]))
                    if rotate == None:
                        pass
                    elif axis in rotate[i]:
                        dataset[axis] = 1 - dataset[axis]
        if exchange_xy:
            self.dataset_B[['x', 'y']] = self.dataset_B[['y', 'x']]

        # print("matching:", matching)
        if not subset is None:
            matching = matching[:, subset]
        if matching.shape[1] > subsample_size > 0:
            self.matching = matching[:, np.random.choice(matching.shape[1], subsample_size, replace=False)]
        else:
            subsample_size = matching.shape[1]
            self.matching = matching
        # print(f'Subsampled {subsample_size} pairs from {matching.shape[1]}')

        self.datasets = [self.dataset_A, self.dataset_B]

    def draw_3D(self,
                size: Optional[List[int]] = [10, 10],
                conf_cutoff: Optional[float] = 0,
                point_size: Optional[List[int]] = [0.1, 0.1],
                line_width: Optional[float] = 0.3,
                line_color: Optional[str] = 'grey',
                line_alpha: Optional[float] = 0.7,
                hide_axis: Optional[bool] = False,
                show_error: Optional[bool] = True,
                show_celltype: Optional[bool] = False,
                cmap: Optional[bool] = 'Reds',
                have_color_map=False,
                color_map=None,
                save: Optional[str] = None
                ) -> None:
        r"""
        Draw 3D picture of two datasets

        Parameters:
        ----------
        size
            plt figure size
        conf_cutoff
            confidence cutoff of mapping to be plotted
        point_size
            point size of every dataset
        line_width
            pair line width
        line_color
            pair line color
        line_alpha
            pair line alpha
        hide_axis
            if hide axis
        show_error
            if show error celltype mapping with different color
        cmap
            color map when vis expr
        save
            save file path
        """
        self.conf_cutoff = conf_cutoff
        show_error = show_error if self.meta else False
        fig = plt.figure(figsize=(size[0], size[1]))
        ax = fig.add_subplot(111, projection='3d')
        # color by meta
        if self.meta:
            color = get_color(len(self.celltypes))
            if have_color_map:
                c_map = color_map
            else:
                c_map = {}
                for i, celltype in enumerate(self.celltypes):
                    c_map[celltype] = color[i]
            if self.expr:
                c_map = cmap
                # expr_concat = pd.concat(self.datasets)[self.expr].to_numpy()
                # norm = plt.Normalize(expr_concat.min(), expr_concat.max())
            for i, dataset in enumerate(self.datasets):
                if self.expr:
                    norm = plt.Normalize(dataset[self.expr].to_numpy().min(), dataset[self.expr].to_numpy().max())
                for cell_type in self.celltypes:
                    slice = dataset[dataset[self.meta] == cell_type]
                    xs = slice['x']
                    ys = slice['y']
                    zs = i
                    if self.expr:
                        ax.scatter(xs, ys, zs, s=point_size[i], c=slice[self.expr], cmap=c_map, norm=norm)
                    else:
                        ax.scatter(xs, ys, zs, s=point_size[i], c=c_map[cell_type])
        # plot points without meta
        else:
            for i, dataset in enumerate(self.datasets):
                xs = dataset['x']
                ys = dataset['y']
                zs = i
                ax.scatter(xs, ys, zs, s=point_size[i])
        # plot line
        self.c_map = c_map
        self.draw_lines(ax, show_error, show_celltype, line_color, line_width, line_alpha)
        if hide_axis:
            plt.axis('off')
        if save != None:
            plt.savefig(save)
        plt.show()

    def draw_lines(self, ax, show_error, show_celltype, line_color, line_width=0.3, line_alpha=0.7) -> None:
        r"""
        Draw lines between paired cells in two datasets
        """
        for i in range(self.matching.shape[1]):
            if not self.conf is None and self.conf[i] < self.conf_cutoff:
                continue
            pair = self.matching[:, i]
            default_color = line_color
            if self.meta != None:
                celltype1 = self.dataset_A.loc[self.dataset_A['index'] == pair[1], self.meta].astype(str).values[0]
                celltype2 = self.dataset_B.loc[self.dataset_B['index'] == pair[0], self.meta].astype(str).values[0]
                if show_error:
                    if celltype1 == celltype2:
                        color = '#ade8f4'  # blue
                    else:
                        color = '#ffafcc'  # red
                if show_celltype:
                    if celltype1 == celltype2:
                        color = self.c_map[celltype1]
                    else:
                        color = '#bbbebf'  # celltype1 error match color
            point0 = np.append(self.dataset_A[self.dataset_A['index'] == pair[1]][['x', 'y']], 0)
            point1 = np.append(self.dataset_B[self.dataset_B['index'] == pair[0]][['x', 'y']], 1)

            coord = np.row_stack((point0, point1))
            color = color if show_error or show_celltype else default_color
            ax.plot(coord[:, 0], coord[:, 1], coord[:, 2], color=color, linestyle="dashed", linewidth=line_width,
                    alpha=line_alpha)


class match_3D_multi_error(match_3D_multi):
    r"""
    Highlight the error mapping between datasets, child of class:`match_3D_multi()`

    Parameters
    ---------
    dataset_A
        pandas dataframe which contain ['index','x','y']
    dataset_B
        pandas dataframe which contain ['index','x','y']
    matching
        matching results
    mode
        which cell pairs to highlight
    highlight_color
        color to highlight the line
    meta
        dataframe colname of meta, such as celltype
    expr
        dataframe colname of gene expr
    subsample_size
        subsample size of matches
    reliability
        if the match is reliable
    scale_coordinate
        if scale the coordinate via `data - np.min(data)) / (np.max(data) - np.min(data))`
    rotate
        how to rotate the slides (force scale_coordinate)
    change_xy
        exchange x and y on dataset_B
    subset
        index of query cells to be plotted

    Note
    ----------
    dataset_A and dataset_B can in different length

    """

    def __init__(self, dataset_A: pd.DataFrame,
                 dataset_B: pd.DataFrame,
                 matching: np.ndarray,
                 mode: Optional[str] = 'high_true',
                 highlight_color: Optional[str] = 'red',
                 meta: Optional[str] = None,
                 expr: Optional[str] = None,
                 subsample_size: Optional[int] = 300,
                 reliability: Optional[np.ndarray] = None,
                 scale_coordinate: Optional[bool] = False,
                 rotate: Optional[List[str]] = None,
                 exchange_xy: Optional[bool] = False,
                 subset: Optional[Union[np.ndarray, List[int]]] = None
                 ) -> None:
        super(match_3D_multi_error, self).__init__(dataset_A, dataset_B, matching, meta, expr, subsample_size,
                                                   reliability,
                                                   scale_coordinate, rotate, exchange_xy, subset)
        assert mode in ['high_true', 'low_true', 'high_false', 'low_false']
        self.mode = mode
        self.highlight_color = highlight_color

    def draw_lines(self, ax, show_error, show_celltype, default_color, line_width=0.3, line_alpha=0.7) -> None:
        for i in range(self.matching.shape[1]):
            pair = self.matching[:, i]
            if self.dataset_B.loc[self.dataset_B['index'] == pair[0], 'celltype'].astype(str).values == \
                    self.dataset_A.loc[self.dataset_A['index'] == pair[1], 'celltype'].astype(str).values:
                if 'false' in self.mode:
                    continue
            if not self.conf is None:
                if 'low' in self.mode and not self.conf[i]:
                    continue
            point0 = np.append(self.dataset_A[self.dataset_A['index'] == pair[1]][['x', 'y']], 0)
            point1 = np.append(self.dataset_B[self.dataset_B['index'] == pair[0]][['x', 'y']], 1)
            coord = np.row_stack((point0, point1))
            ax.scatter(point0[0], point0[1], point0[2], color='red', alpha=1, s=0.3)
            ax.scatter(point1[0], point1[1], point1[2], color='red', alpha=1, s=0.3)
            ax.plot(coord[:, 0], coord[:, 1], coord[:, 2], color=self.highlight_color,
                    linestyle="dashed", linewidth=line_width, alpha=line_alpha)


class match_3D_celltype(match_3D_multi):
    r"""
    Highlight the celltype mapping, child of class:`match_3D_multi()`

    Parameters
    ---------
    dataset_A
        pandas dataframe which contain ['index','x','y']
    dataset_B
        pandas dataframe which contain ['index','x','y']
    matching
        matching results
    highlight_celltype
        celltypes to highlight in two datasets
    highlight_line
        color to highlight the line
    highlight_cell
        color to highlight the cell
    meta
        dataframe colname of meta, such as celltype
    expr
        dataframe colname of gene expr
    subsample_size
        subsample size of matches
    reliability
        if the match is reliable
    scale_coordinate
        if scale the coordinate via `data - np.min(data)) / (np.max(data) - np.min(data))`
    rotate
        how to rotate the slides (force scale_coordinate)
    change_xy
        exchange x and y on dataset_B
    subset
        index of query cells to be plotted

    Note
    ----------
    dataset_A and dataset_B can in different length
    """

    def __init__(self, dataset_A: pd.DataFrame,
                 dataset_B: pd.DataFrame,
                 matching: np.ndarray,
                 highlight_celltype: Optional[List[List[str]]] = [[], []],
                 highlight_line: Optional[Union[List[str], str]] = 'red',
                 highlight_cell: Optional[str] = None,
                 meta: Optional[str] = None,
                 expr: Optional[str] = None,
                 subsample_size: Optional[int] = 300,
                 reliability: Optional[np.ndarray] = None,
                 scale_coordinate: Optional[bool] = False,
                 rotate: Optional[List[str]] = None,
                 exchange_xy: Optional[bool] = False,
                 subset: Optional[Union[np.ndarray, List[int]]] = None
                 ) -> None:
        super(match_3D_celltype, self).__init__(dataset_A, dataset_B, matching, meta, expr, subsample_size,
                                                reliability, scale_coordinate, rotate, exchange_xy, subset)
        assert set(highlight_celltype[0]).issubset(set(self.celltypes))
        assert set(highlight_celltype[1]).issubset(set(self.celltypes))
        self.highlight_celltype = highlight_celltype
        self.highlight_line = highlight_line
        self.highlight_cell = highlight_cell

    def draw_lines(self, ax, show_error, show_celltype, default_color, line_width: float = 0.3,
                   line_alpha: float = 0.7) -> None:
        if len(self.highlight_celltype[0]) >= len(self.highlight_celltype[1]):
            color_index = self.highlight_celltype[0]
        else:
            color_index = self.highlight_celltype[1]
        if type(self.highlight_line) == list and len(self.highlight_line) >= len(color_index):
            cmap = self.highlight_line
        else:
            cmap = get_color(len(color_index))

        for i in range(self.matching.shape[1]):
            pair = self.matching[:, i]
            a = self.dataset_A.loc[self.dataset_A['index'] == pair[1], self.meta].astype(str).values
            b = self.dataset_B.loc[self.dataset_B['index'] == pair[0], self.meta].astype(str).values
            if a not in self.highlight_celltype[0] or b not in self.highlight_celltype[1]:
                continue
            point0 = np.append(self.dataset_A[self.dataset_A['index'] == pair[1]][['x', 'y']], 0)
            point1 = np.append(self.dataset_B[self.dataset_B['index'] == pair[0]][['x', 'y']], 1)
            coord = np.row_stack((point0, point1))
            if self.highlight_cell:
                ax.scatter(point0[0], point0[1], point0[2], color=self.highlight_cell, alpha=1, s=1)
                ax.scatter(point1[0], point1[1], point1[2], color=self.highlight_cell, alpha=1, s=1)
            if isinstance(cmap, list):
                color = cmap[color_index.index(a)] if len(self.highlight_celltype[0]) >= len(
                    self.highlight_celltype[1]) else cmap[color_index.index(b)]
            else:
                color = cmap
            color = color if show_error else self.highlight_line
            ax.plot(coord[:, 0], coord[:, 1], coord[:, 2], color=color, linestyle="dashed", linewidth=line_width,
                    alpha=line_alpha)


def Sankey(matching_table: pd.DataFrame,
           filter_num: Optional[int] = 50,
           color: Optional[List[str]] = 'red',
           title: Optional[str] = '',
           prefix: Optional[List[str]] = ['E11.5', 'E12.5'],
           layout: Optional[List[int]] = [1300, 900],
           font_size: Optional[float] = 15,
           font_color: Optional[str] = 'Black',
           save_name: Optional[str] = None,
           format: Optional[str] = 'png',
           width: Optional[int] = 1200,
           height: Optional[int] = 1000,
           return_fig: Optional[bool] = False
           ) -> None:
    r"""
    Sankey plot of celltype

    Parameters
    ----------
    matching_tables
        list of matching table
    filter_num
        filter number of matches
    color
        color of node
    title
        plot title
    prefix
        prefix to distinguish datasets
    layout
        layout size of picture
    font_size
        font size in plot
    font_color
        font color in plot
    save_name
        save file name (None for not save)
    format
        save picture format (see https://plotly.com/python/static-image-export/ for more details)
    width
        save picture width
    height
        save picture height
    return_fig
        if return plotly figure
    """
    source, target, value = [], [], []
    label_ref = [a + f'_{prefix[0]}' for a in matching_table.columns.to_list()]
    label_query = [a + f'_{prefix[1]}' for a in matching_table.index.to_list()]
    label_all = label_query + label_ref
    label2index = dict(zip(label_all, list(range(len(label_all)))))

    for i, query in enumerate(label_query):
        for j, ref in enumerate(label_ref):
            if int(matching_table.iloc[i, j]) > filter_num:
                target.append(label2index[query])
                source.append(label2index[ref])
                value.append(int(matching_table.iloc[i, j]))

    fig = go.Figure(
        data=[go.Sankey(
            node=dict(pad=50,
                      thickness=50,
                      line=dict(color="green", width=0.5),
                      label=label_all,
                      color=color),
            link=dict(source=source,  # indices correspond to labels, eg A1, A2, A1, B1, ...
                      target=target,
                      value=value)
        )
        ],
        layout=go.Layout(autosize=False, width=layout[0], height=layout[1])
    )

    fig.update_layout(title_text=title, font_size=font_size, font_color=font_color)
    fig.show()
    if save_name != None:
        fig.write_image(save_name + f'.{format}', width=width, height=height)
    if return_fig:
        return fig


def multi_Sankey(matching_tables: List[pd.DataFrame],
                 color: Optional[List[str]] = 'random',
                 title: Optional[str] = 'Sankey plot',
                 layout: Optional[List[int]] = [1300, 900],
                 day: Optional[float] = 0,
                 save_name: Optional[str] = None,
                 format: Optional[str] = 'svg',
                 ) -> None:
    r"""
    Sankey plot of celltype in multi datasets

    Parameters
    ----------
    matching_tables
        list of matching table
    color
        how to color the nodes, 'random' for random color, 'celltype' for color by celltype
    title
        plot title
    layout
        layout size of picture
    day
        start day of dataset for temporal order
    save_name
        save file name (None for not save)
    format
        saved picture format (see https://plotly.com/python/static-image-export/ for more details)
    """
    mappings = len(matching_tables) + 1
    prefixes = [day + i for i in range(mappings)]
    source, target, value, label_all = [], [], [], set()
    for i, matching_table in enumerate(matching_tables):
        label_ref = [a + f'_{prefixes[i]}' for a in matching_table.columns.to_list()]
        label_query = [a + f'_{prefixes[i + 1]}' for a in matching_table.index.to_list()]
        # label_all.add(label_ref)
        for i in label_ref + label_query:
            label_all.add(i)
    label2index = dict(zip(label_all, list(range(len(label_all)))))

    for matching_table, prefix in zip(matching_tables, prefixes):
        for i, query in enumerate(matching_table.index):
            for j, ref in enumerate(matching_table.columns):
                if int(matching_table.iloc[i, j]) > 10:
                    target.append(label2index[query + '_' + str(prefix + 1)])
                    source.append(label2index[ref + '_' + str(prefix)])
                    value.append(int(matching_table.iloc[i, j]))

    if color == 'random':
        color = [get_color()] * matching_tables[0].shape[0]
        for matching_table in matching_tables:
            color += [get_color()] * matching_table.shape[1]
    elif color == 'celltype':
        pass

    fig = go.Figure(
        data=[go.Sankey(node=dict(
            pad=50,
            thickness=50,
            line=dict(color="green", width=0.5),
            label=list(label_all),
            color=color
        ),
            link=dict(source=source,  # indices correspond to labels, eg A1, A2, A1, B1, ...
                      target=target,
                      value=value)
        )
        ],
        layout=go.Layout(autosize=False, width=layout[0], height=layout[1])
    )

    fig.update_layout(title_text=title, font_size=10)
    if save_name != None:
        fig.write_image(save_name + f'.{format}', width=layout[0], height=layout[1])
    fig.show()


def matching_2d(matching: np.ndarray,
                ref: AnnData,
                src: AnnData,
                biology_meta: str,
                topology_meta: str,
                spot_size: Optional[int] = 5,
                title: Optional[str] = "2D matching",
                save: Optional[str] = None
                ) -> None:
    r"""
    Visualize the matching result in 2D space

    Parameters
    ----------
    matching
        matching result
    ref
        reference dataset
    src
        target dataset
    biology_meta
        celltype meta colname of adata.obs
    topology_meta
        region meta colname of adata.obs
    spot_size
        size of spot for visualization
    title
        plot title
    save
        save file name (None for not save)
    """
    src.obs['target_celltype'] = ref.obs.iloc[matching[1, :], :][biology_meta].to_list()
    src.obs['target_region'] = ref.obs.iloc[matching[1, :], :][topology_meta].to_list()
    src.obs["vis"] = 'celltype_false_region_false'
    src.obs["vis"] = src.obs["vis"].astype('str')

    cell_type_match = src.obs[biology_meta] == src.obs['target_celltype']
    region_match = src.obs[topology_meta] == src.obs['target_region']
    cell_type_match = cell_type_match.to_numpy()
    region_match = region_match.to_numpy()

    src.obs.loc[np.logical_and(cell_type_match, region_match), 'vis'] = 'celltype_true_region_true'
    src.obs.loc[np.logical_and(~cell_type_match, region_match), 'vis'] = 'celltype_false_region_true'
    src.obs.loc[np.logical_and(cell_type_match, ~region_match), 'vis'] = 'celltype_true_region_false'

    sc.pl.spatial(src, color="vis", spot_size=spot_size, title=title,
                  palette=['red', 'purple', 'yellow', 'green'],
                  save=save)

    del src.obs['target_celltype']
    del src.obs['target_region']
    del src.obs['vis']


def plot_aligned_slice(source_slice_name, target_slice_name,
                       source_data, target_data,
                       source_cluster_index, target_cluster_index,
                       target_aligned_section, target_aligned_all):
    cluster_method = 'louvain'
    plt.rcParams["figure.figsize"] = (3, 3)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3),
                           gridspec_kw={'wspace': 0.5, 'hspace': 0.1})
    ax[0].scatter(source_data.obsm["spatial"][:, 0],
                  source_data.obsm["spatial"][:, 1],
                  c="blue", cmap=plt.cm.binary_r, s=1)
    ax[0].scatter(source_data[source_data.obs[cluster_method].isin([source_cluster_index])].obsm["spatial"][:, 0],
                  source_data[source_data.obs[cluster_method].isin([source_cluster_index])].obsm["spatial"][:, 1],
                  c="red", cmap=plt.cm.binary_r, s=1)
    ax[0].set_title('Source ' + source_slice_name, size=14)

    ax[1].scatter(target_data.obsm["spatial"][:, 0],
                  target_data.obsm["spatial"][:, 1],
                  c="blue", cmap=plt.cm.binary_r, s=1)
    ax[1].scatter(target_data[target_data.obs[cluster_method].isin([target_cluster_index])].obsm["spatial"][:, 0],
                  target_data[target_data.obs[cluster_method].isin([target_cluster_index])].obsm["spatial"][:, 1],
                  c="red", cmap=plt.cm.binary_r, s=1)
    ax[1].set_title('Target origin ' + target_slice_name, size=14)

    ax[2].scatter(target_aligned_all[:, 0], target_aligned_all[:, 1],
                  c="blue", cmap=plt.cm.binary_r, s=1)
    ax[2].scatter(target_aligned_section[:, 0], target_aligned_section[:, 1],
                  c="red", cmap=plt.cm.binary_r, s=1)
    ax[2].set_title('Target aligned ' + target_slice_name, size=14)

    plt.axis("equal")
    # plt.axis("off")
    plt.show()


def plot_single_slice(adata, spatial_key, cluster_key, frameon=False, i=1, j=1, n=1, s=1):
    ind = np.sort(adata.obs[cluster_key].unique().copy())
    color = adata.uns[cluster_key + '_colors'].copy()
    dic = dict(zip(ind, color))
    col = adata.obs[cluster_key].replace(dic)
    ax = plt.subplot(i, j, n)
    if type(adata.obsm[spatial_key]) == pd.core.frame.DataFrame:
        plt.scatter(adata.obsm[spatial_key].iloc[:, 0], adata.obsm[spatial_key].iloc[:, 1], c=col, s=s, rasterized=True)
    if type(adata.obsm[spatial_key]) == np.ndarray:
        plt.scatter(adata.obsm[spatial_key][:, 0], adata.obsm[spatial_key][:, 1], c=col, s=s, rasterized=True)
    plt.axis('equal')
    if not frameon:
        plt.axis('off')


def plot_slices_loc(ad_list, spatial_key, cluster_key,
                    legend=True, frameon=False, colors=None,
                    i=1, j=1, s=1):
    clusters = []
    colored_num = 0
    for ad in ad_list:
        if f'{cluster_key}_colors' in ad.uns.keys():
            colored_num += 1
        clusters.extend(ad.obs[cluster_key].cat.categories)
    clusters = np.unique(clusters)
    if colored_num < len(ad_list):
        if colors is None:
            if len(clusters) > 10:
                colors = [matplotlib.colors.to_hex(c) for c in sns.color_palette('tab20', n_colors=len(clusters))]
            else:
                colors = [matplotlib.colors.to_hex(c) for c in sns.color_palette('tab10', n_colors=len(clusters))]
        color_map = pd.DataFrame(colors, index=clusters, columns=['color'])
        for ad in ad_list:
            ad.uns[f'{cluster_key}_color_map'] = color_map
            ad.uns[f'{cluster_key}_colors'] = [color_map.loc[c, 'color'] for c in ad.obs[cluster_key].cat.categories]
    else:
        color_map = pd.DataFrame(index=clusters, columns=['color'])
        for ad in ad_list:
            color_map.loc[ad.obs[cluster_key].cat.categories, 'color'] = ad.uns[f'{cluster_key}_colors']
            ad.uns[f'{cluster_key}_color_map'] = color_map

    for k in range(len(ad_list)):
        plot_single_slice(ad_list[k], spatial_key, cluster_key, frameon, i, j, n=np.min([k + 1, i * j]), s=s)

    if legend:
        legend_elements = [
            Line2D([], [], marker='.', markersize=10, color=color_map.loc[i, 'color'], linestyle='None', label=i) for i
            in color_map.index]
        plt.legend(handles=legend_elements, loc='right')


def plot_stacked_slices_3d(adata_list,
                           adata_name_list,
                           aligned_points_dict,
                           cluster_num=20,
                           cluster_method='mclust',
                           figsize=(8, 8),
                           elev=None,
                           azim=None,
                           frameon=True,
                           alpha=1,
                           save=None,
                           have_color_map=False,
                           color_map=None,
                           use_origin_loc=False,
                           ):
    """Plot all slices stacked in 3D

    Plot all slices stacked with a given number of subplots rows and columns. Spots/cells colored by spatial domain.
    """

    coo = pd.DataFrame(adata_list[0].obsm['spatial'].copy(), columns=['X', 'Y'])
    coo['Z'] = 0

    if have_color_map:
        coo['domain_colors'] = list(adata_list[0].obs[cluster_method])
    else:
        coo['domain_colors'] = list(adata_list[0].obs[cluster_method].replace(dict(zip(list(range(cluster_num)),
                                                                                    get_color(cluster_num)))))

    for i in range(1, len(adata_name_list)):
        loc = pd.DataFrame(aligned_points_dict[adata_name_list[i]].copy(), columns=['X', 'Y'])
        loc['Z'] = i

        if have_color_map:
            loc['domain_colors'] = list(adata_list[i].obs[cluster_method])
        else:
            loc['domain_colors'] = list(adata_list[i].obs[cluster_method].replace(dict(zip(list(range(cluster_num)),
                                                                                        get_color(cluster_num)))))
        
        coo = pd.concat([coo, loc], axis=0)

    loc_3d = coo.values[:, :3]
    color = coo['domain_colors']
    if have_color_map:
        temp = color.copy()
        color = [color_map[c] for c in temp]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    if (elev is not None) and (azim is not None):
        ax.view_init(elev, azim)

    ax.scatter(loc_3d[:, 0], loc_3d[:, 1], loc_3d[:, 2], c=color, alpha=alpha)


    ax.set_xticks([]);
    ax.set_yticks([]);
    ax.set_zticks([]);

    if not frameon:
        plt.axis('off')

    if save != None:
        plt.savefig(save)

    plt.show()

