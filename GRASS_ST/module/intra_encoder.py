import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.spmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)

class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
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
        # print("mp ", beta.data.cpu().numpy())  # semantic attention
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i]*beta[i]
        return z_mp


class Intra_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_list, attn_drop):
        super(Intra_encoder, self).__init__()
        self.P = len(hidden_dim_list)
        self.en_hid_dim = [input_dim] + hidden_dim_list
        self.de_hid_dim = self.en_hid_dim[::-1]
        self.gcn_encoder = nn.ModuleList([GCN(self.en_hid_dim[i], self.en_hid_dim[i+1]) for i in range(self.P)])
        self.gcn_decoder = nn.ModuleList([GCN(self.de_hid_dim[i], self.de_hid_dim[i+1]) for i in range(self.P)])

        # self.att = Attention(hidden_dim_list[-1], attn_drop)

    def forward(self, h, shuf_h, mps):
        # z_mp = self.gcn(h, mps[0])
        en_emb_list = []
        en_shuf_emb_list = []
        de_recon_list = []
        en_embed = h
        en_shuf_embed = shuf_h
        for i in range(self.P):
            en_embed = self.gcn_encoder[i](en_embed, mps[0])
            en_shuf_embed = self.gcn_encoder[i](en_shuf_embed, mps[0])
            en_emb_list.append(en_embed)
            en_shuf_emb_list.append(en_shuf_embed)

        de_recon = en_embed
        for i in range(self.P):
            de_recon = self.gcn_decoder[i](de_recon, mps[0])
            de_recon_list.append(de_recon)

        return en_embed, en_emb_list, en_shuf_embed, en_shuf_emb_list, de_recon, de_recon_list
