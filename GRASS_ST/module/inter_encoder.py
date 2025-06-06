import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class inter_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(inter_att, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        # print("sc ", beta.data.cpu().numpy())  # type-level attention
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc


class intra_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(intra_att, self).__init__()
        self.att = nn.Parameter(torch.empty(size=(1, 2*hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        self.softmax = nn.Softmax(dim=1)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, nei, h, h_refer):
        nei_emb = F.embedding(nei, h)
        h_refer = torch.unsqueeze(h_refer, 1)
        h_refer = h_refer.expand_as(nei_emb)
        all_emb = torch.cat([h_refer, nei_emb], dim=-1)
        attn_curr = self.attn_drop(self.att)
        att = self.leakyrelu(all_emb.matmul(attn_curr.t()))
        att = self.softmax(att)
        nei_emb = (att*nei_emb).sum(dim=1)
        return nei_emb


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, sample_rate, nei_num, attn_drop, 
                 bias=True, device=None):
        super(Expert, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        self.act = nn.PReLU()
        self.device = device

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(hidden_dim))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        self.intra = nn.ModuleList([intra_att(hidden_dim, attn_drop) for _ in range(nei_num)])
        self.inter = inter_att(hidden_dim, attn_drop)
        self.sample_rate = sample_rate
        self.nei_num = nei_num

    def forward(self, nei_h, nei_index, index=0, 
                num_samples_inter=None):
        nei_z = []
        for i in range(len(nei_h)):
            nei_z_tmp = self.act(self.fc(nei_h[i]))
            nei_z.append(nei_z_tmp)
        embeds = []

        if num_samples_inter is not None and num_samples_inter < self.nei_num:
            indices = np.random.choice(list(range(self.nei_num)), num_samples_inter, replace=False)
        else:
            indices = list(range(self.nei_num))
        
        # print("inter indices:", indices)

        for i in indices:
            sele_nei = []
            sample_num = self.sample_rate[i]
            for per_node_nei in nei_index[i]:
                if len(per_node_nei) >= sample_num:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,
                                                               replace=False))[np.newaxis]
                else:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,
                                                               replace=True))[np.newaxis]
                sele_nei.append(select_one)
            
            # print("Expert device:", self.device)
            sele_nei = torch.cat(sele_nei, dim=0).to(self.device)
            # print(f"i:{i}, index:{index}")
            if index==0:
                one_type_emb = F.elu(self.intra[i](sele_nei, nei_z[i + 1], nei_z[index]))
            else:
                if i<index:
                    one_type_emb = F.elu(self.intra[i](sele_nei, nei_z[i], nei_z[index]))
                else:
                    one_type_emb = F.elu(self.intra[i](sele_nei, nei_z[i+1], nei_z[index]))

            embeds.append(one_type_emb)
        z_mc = self.inter(embeds)
        return z_mc
