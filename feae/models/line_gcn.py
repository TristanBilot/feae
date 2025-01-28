import copy

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


class LineGCN(nn.Module):
    def __init__(self, ndim_in, hid_size, ndim_out, residual, norm):
        super(LineGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.residual = residual
        # two-layer GCN
        self.layers.append(
            GraphConv(ndim_in, hid_size, activation=F.relu, norm=norm)
        )
        self.layers.append(GraphConv(hid_size, ndim_out, norm=norm))
        self.dropout = nn.Dropout(0.)
        self.res_fc = nn.Linear(ndim_in, ndim_out, bias=False)
        self.mlp = nn.Linear(hid_size, 2)

    def forward(self, g, node_feats, corrupt=False, get_embeddings=False):
        res = copy.deepcopy(node_feats)
        h = node_feats
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        
        h = h.squeeze()
        if self.residual:
            res = self.res_fc(res)
            h = h + res

        if get_embeddings:
            return None, h
        return None, self.mlp(h)
