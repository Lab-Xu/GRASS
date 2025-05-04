
import torch
import torch.nn as nn
import torch.nn.functional as F

class Feature_map(nn.Module):
    def __init__(self, input_dim, hidden_dim, sample_rate, nei_num, attn_drop, 
                 bias=True, device=None):
        super(Feature_map, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        self.act = nn.PReLU()
        self.device = device

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(hidden_dim))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

    def forward(self, feat):
        P_matrix = self.act(self.fc(feat))
        return P_matrix
