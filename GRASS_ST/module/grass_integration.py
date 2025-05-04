import torch
import torch.nn as nn
import torch.nn.functional as F
from .intra_encoder import Intra_encoder
from .inter_encoder import Expert
from .intra_contrast import Intra_Contrast
from .inter_contrast import Inter_Contrast
from .feature_map import Feature_map
import numpy as np

class GRASS_Integration(nn.Module):
    def __init__(self, hidden_dim_list, feat_dim, mapping_dim, feat_drop, attn_drop, P, sample_rate,
                 nei_num, tau, lam, use_mix_expert=False, w_neg=0.5,device=None):
        # print("P:", P)
        super(GRASS_Integration, self).__init__()
        self.hidden_dim = hidden_dim_list[-1]
        self.input_dim = feat_dim
        self.mapping_dim = mapping_dim
        self.use_mix_expert = use_mix_expert
        self.w_neg = w_neg

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

        self.intra_list =nn.ModuleList([Intra_encoder(self.mapping_dim, hidden_dim_list, attn_drop)])

        if self.use_mix_expert:
            self.inter_shared_list = nn.ModuleList([Expert(self.mapping_dim, self.hidden_dim, sample_rate, nei_num, attn_drop,
                                                    device=device)])

        self.inter_ind_list = nn.ModuleList([Expert(self.mapping_dim, self.hidden_dim, sample_rate, nei_num, attn_drop,
                                                device=device)
                                                for _ in range(nei_num+1)])
        self.intra_contrast = Intra_Contrast(self.hidden_dim)
        self.inter_contrast = Inter_Contrast(self.hidden_dim, tau, lam)


    def forward(self, feats, shuf_feats, lbl_list,
                pos_list, 
                mps_list, nei_index_list,
                w_intra=1, w_re=1, w_inter=1,
                num_samples=None,
                num_samples_inter=None):
        
        loss_all = 0

        mapping_feats = feats
        mapping_shuf_feats = shuf_feats
        num_slices = len(mapping_feats)
        # print("mapping_feats:", mapping_feats)


        if num_samples is not None and num_samples < num_slices:
            indices = np.random.choice(list(range(num_slices)), num_samples, replace=False)
        else:
            indices = range(num_slices)

        # print("all indices:", indices)

        for i in indices:
            lbl = lbl_list[i]
            h_emb, _, h_shuf_emb, _, x_recon, _ = self.intra_list[0](mapping_feats[i], mapping_shuf_feats[i], mps_list[i])

            if self.use_mix_expert:
                z_emb_shared = self.inter_shared_list[0](mapping_feats, nei_index_list[i], index=i, num_samples_inter=num_samples_inter)
                z_emb_ind = self.inter_ind_list[i](mapping_feats, nei_index_list[i], index=i, num_samples_inter=num_samples_inter)
                z_emb = self.w_neg*z_emb_shared + (1-self.w_neg)*z_emb_ind
            else:
                z_emb = self.inter_ind_list[i](mapping_feats, nei_index_list[i], index=i, num_samples_inter=num_samples_inter)
            loss_intra = self.intra_contrast(h_emb, h_shuf_emb, lbl)
            loss_inter = self.inter_contrast(h_emb, z_emb, pos_list[i])
            loss_recon = F.mse_loss(x_recon, feats[i])
            loss_all += w_intra*loss_intra + w_re*loss_recon + w_inter*loss_inter

        return loss_all

    def get_embeds(self, feats, mps_list):
        emb_list = []
        reX_list = []
        for i in range(len(feats)):
            h_emb, _, _, _, x_recon, _ = self.intra_list[0](feats[i], feats[i], mps_list[i])
            emb_list.append(h_emb.detach())
            reX_list.append(x_recon.detach())
        combined_emb = torch.cat(emb_list, dim=0)
        combined_reX = torch.cat(reX_list, dim=0)
        return combined_emb, emb_list, combined_reX, reX_list
