import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

from GRASS_ST.module.ae import New_NB_AE, New_ZINB_AE


def nb_loss(data, mean, disp):
    eps = 1e-10
    loss1 = torch.lgamma(disp + eps) + torch.lgamma(data + 1) - torch.lgamma(data + disp + eps)
    loss2 = (disp + data) * torch.log(1.0 + (mean / (disp + eps))) + (
            data * (torch.log(disp + eps) - torch.log(mean + eps)))
    return loss1 + loss2


def zinb_loss(data, mean, disp, drop, ridge_lambda=0):
    eps = 1e-10
    nb_case = nb_loss(data, mean, disp) - torch.log(1.0 - drop + eps)
    zero_nb = torch.pow(disp / (disp + mean + eps), disp)
    zero_case = -torch.log(drop + ((1.0 - drop) * zero_nb) + eps)
    result = torch.where(torch.lt(data, 1e-10), zero_case, nb_case)
    ridge = ridge_lambda * torch.pow(drop, 2)
    result += ridge
    return result.mean()


def nll_loss(data, mean, disp, drop=None, dist='zinb'):
    if dist == 'nb':
        return nb_loss(data, mean, disp).mean()
    else:
        return zinb_loss(data, mean, disp, drop, ridge_lambda=0)


class MyDataset(Dataset):
    '''
    construct dataset of model1
    input: adata
    output: Dataset with feature, count and library size.
    '''

    def __init__(self, adata, count_key=None, size='explog', batch_key=None):
        super(MyDataset, self).__init__()

        if count_key is None:
            count_key = 'counts'
            adata.layers['counts'] = adata.X.copy()

        # set count matrix
        count = adata.layers[count_key].copy()

        if sp.issparse(count):
            self.count = torch.from_numpy(count.toarray()).float()
        else:
            self.count = torch.from_numpy(count).float()

            ### library size vector
        if size == 'explog':
            self.size = torch.exp(torch.log10(self.count.sum(axis=1))).unsqueeze(1)
        elif size == 'sum':
            self.size = self.count.sum(axis=1).unsqueeze(1)
        elif size == 'median':
            self.size = (self.count.sum(1) / np.median(self.count.sum(1))).unsqueeze(1)
        adata.obs['size'] = self.size.squeeze(1).numpy()

        ### count matrix
        if sp.issparse(adata.layers[count_key]):
            self.count = torch.from_numpy(adata.layers[count_key].toarray()).float()
        else:
            self.count = torch.from_numpy(adata.layers[count_key]).float()

            ### input feature matrix
        if sp.issparse(adata.X):
            self.feature = torch.from_numpy(adata.X.toarray()).float()
        else:
            self.feature = torch.from_numpy(adata.X).float()

            ### batch labels
        if batch_key is not None:
            # self.batch = torch.from_numpy(adata.obs[batch_key].values)
            self.batch = torch.from_numpy(pd.get_dummies(adata.obs[batch_key]).values).float()
            self.all_data = [(self.feature[i], self.count[i], self.size[i], self.batch[i]) for i in
                             range(self.feature.shape[0])]

        else:
            self.all_data = [(self.feature[i], self.count[i], self.size[i]) for i in range(self.feature.shape[0])]

    def __getitem__(self, idx):
        return self.all_data[idx]

    def __len__(self):
        return len(self.all_data)


class AE_Model(object):

    def __init__(
            self,
            adata,
            batch_key=None,
            n_hidden: int = 128,
            n_latent: int = 32,
            dropout_rate: float = 0.2,
            likelihood: str = "nb",
            device: str = None,
            num_workers: int = 4,
            result_path=None,
    ):
        super(AE_Model, self).__init__()

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device =  torch.device('cuda:0') if torch.cuda.is_available() else  torch.device('cpu')

        self.adata = adata
        self.n_batch = None if batch_key is None else len(set(adata.obs[batch_key]))
        # print("self.n_batch:", self.n_batch)
        self.batch_key = batch_key

        self.n_input = adata.shape[1]

        self.n_latent = n_latent
        self.likelihood = likelihood

        ae_dict = {'zinb': New_ZINB_AE, 'nb': New_NB_AE}

        self.ae = ae_dict[self.likelihood](input_dim=self.n_input,
                                           hidden_dim=n_hidden,
                                           latent_dim=n_latent,
                                           dropout=dropout_rate,
                                           n_batch=self.n_batch
                                           ).to(self.device)

        self.num_workers = num_workers
        self.result_path = result_path

    def get_latent(self,
                   batch_size=None,
                   return_data=False
                   ):
        '''
        Preprocessed predicting part using AE module

        Parameters
        ----------
        batch_size
            Batch size in predicting part.
        return_data
            Whether to return adata, default is False.
        '''
        self.ae.eval()

        if batch_size is None:
            batch_size = len(self.data_ae)
        print("batch_size:", batch_size)

        dataloader = DataLoader(self.data_ae, shuffle=False, batch_size=batch_size, num_workers=self.num_workers,
                                drop_last=True)

        z = torch.empty(size=[0, self.n_latent])
        mean = torch.empty(size=[0, self.n_input])

        with torch.no_grad():

            if self.batch_key is not None:
                for _, (feat_tmp, count_tmp, lib_tmp, batch_tmp) in enumerate(dataloader):
                    # print(i)
                    feat_tmp = feat_tmp.to(self.device)
                    count_tmp = count_tmp.to(self.device)
                    lib_tmp = lib_tmp.to(self.device)
                    batch_tmp = batch_tmp.to(self.device)
                    rate_scaled_tmp, logits_tmp, _, z_tmp = self.ae(feat_tmp, batch_tmp)
                    z = torch.cat([z, z_tmp.cpu()[:, :self.n_latent]])
                    rate_tmp = rate_scaled_tmp * lib_tmp
                    mean_tmp = rate_tmp * logits_tmp
                    mean = torch.cat([mean, mean_tmp.cpu()])
            else:
                for _, (feat_tmp, count_tmp, lib_tmp) in enumerate(dataloader):
                    # print(i)
                    feat_tmp = feat_tmp.to(self.device)
                    count_tmp = count_tmp.to(self.device)
                    lib_tmp = lib_tmp.to(self.device)
                    rate_scaled_tmp, logits_tmp, _, z_tmp = self.ae(feat_tmp)
                    z = torch.cat([z, z_tmp.cpu()[:, :self.n_latent]])
                    rate_tmp = rate_scaled_tmp * lib_tmp
                    mean_tmp = rate_tmp * logits_tmp
                    mean = torch.cat([mean, mean_tmp.cpu()])

        self.adata.obsm['latent'] = z.detach().cpu().numpy()
        self.adata.layers['Denoise'] = mean.detach().cpu().numpy()

        if return_data:
            return self.adata

    def run_algorithm(self,
                      count_key=None,
                      lib_size='explog',
                      lr=0.001,
                      weight_decay=0,
                      epoch_ae=100,
                      batch_size=64):

        self.data_ae = MyDataset(self.adata,
                                 count_key=count_key,
                                 size=lib_size,
                                 batch_key=self.batch_key)

        data_loader = DataLoader(self.data_ae,
                                 shuffle=True,
                                 batch_size=batch_size,
                                 num_workers=self.num_workers,
                                 drop_last=True)
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr, weight_decay=weight_decay)

        train_loss = []
        for epoch in tqdm(range(0, epoch_ae)):
            loss_tmp = 0
            if self.batch_key is not None:
                for i, (feat_tmp, count_tmp, size_tmp, batch_tmp) in enumerate(data_loader):
                    # print("feat_tmp shape:", feat_tmp.shape)
                    # print("count_tmp shape:", count_tmp.shape)
                    # print("size_tmp shape:", size_tmp.shape)
                    # print("batch_tmp shape:", batch_tmp.shape)

                    feat_tmp = feat_tmp.to(self.device)
                    count_tmp = count_tmp.to(self.device)
                    size_tmp = size_tmp.to(self.device)
                    batch_tmp = batch_tmp.to(self.device)
                    self.ae.train()
                    rate_scaled_tmp, logits_tmp, drop_tmp, _ = self.ae(feat_tmp, batch_tmp)
                    rate_tmp = rate_scaled_tmp * size_tmp
                    mean_tmp = rate_tmp * logits_tmp
                    optimizer.zero_grad()
                    loss_train = nll_loss(count_tmp, mean_tmp, rate_tmp, drop_tmp, dist=self.likelihood).mean()
                    loss_train.backward()
                    optimizer.step()
                    # if i%5 == 0:
                    # print("AE Epoch:{},  loss {}".format(epoch,loss_train.item()))
                    loss_tmp += loss_train.item()
                train_loss.append(loss_tmp / len(data_loader))
            else:
                for i, (feat_tmp, count_tmp, size_tmp) in enumerate(data_loader):
                    feat_tmp = feat_tmp.to(self.device)
                    count_tmp = count_tmp.to(self.device)
                    size_tmp = size_tmp.to(self.device)
                    self.ae.train()
                    rate_scaled_tmp, logits_tmp, drop_tmp, _ = self.ae(feat_tmp)
                    rate_tmp = rate_scaled_tmp * size_tmp
                    mean_tmp = rate_tmp * logits_tmp
                    optimizer.zero_grad()
                    loss_train = nll_loss(count_tmp, mean_tmp, rate_tmp, drop_tmp, dist=self.likelihood).mean()
                    loss_train.backward()
                    optimizer.step()
                    # if i%5 == 0:
                    # print("AE Epoch:{},  loss {}".format(epoch,loss_train.item()))
                    loss_tmp += loss_train.item()
                train_loss.append(loss_tmp / len(data_loader))

        self.ae.eval()
