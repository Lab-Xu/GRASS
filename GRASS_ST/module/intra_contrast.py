import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        # print("h_pl shape:", h_pl.shape)
        # print("h_mi shape:", h_mi.shape)
        # print("c_x shape:", c_x.shape)

        sc_1 = self.f_k(h_pl, c_x).T
        sc_2 = self.f_k(h_mi, c_x).T

        # print("sc_1 shape:", sc_1.shape)
        # print("sc_2 shape:", sc_2.shape)

        # sc_1 = torch.squeeze(sc_1, 1)  # 移除第 1 维，结果形状: [batch_size]
        # sc_2 = torch.squeeze(sc_2, 1)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)
        
class Intra_Contrast(nn.Module):
    def __init__(self, hidden_dim):
        super(Intra_Contrast, self).__init__()
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(hidden_dim)
        self.b_xent = nn.BCEWithLogitsLoss()

    def forward(self, h_emb, h_shuf_emb, lbl):
        c = self.read(h_emb)
        c = self.sigm(c)
        ret = self.disc(c, h_emb, h_shuf_emb)
        # print("ret shape:", ret.shape)
        # print("lbl shape:", lbl.shape)
        loss = self.b_xent(ret, lbl)
        return loss
