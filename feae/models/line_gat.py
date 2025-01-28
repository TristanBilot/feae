import dgl
import dgl.function as fn
from dgl.nn.pytorch import GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy



#%%
class LineGAT(nn.Module):
    def __init__(self, ndim_in, hid_size, ndim_out, residual, heads=[2, 2]):
        super(LineGAT, self).__init__()
        # self.layers = nn.ModuleList()
        # self.layers.append(GATConv(ndim_in, ndim_out, num_heads=2, residual=residual))
        self.gat_layers = nn.ModuleList()
        self.dropout = nn.Dropout(0.3)
        self.residual = residual
        # two-layer GAT
        self.gat_layers.append(
            GATConv(
                ndim_in,
                hid_size,
                heads[0],
                feat_drop=0.,
                attn_drop=0.,
                activation=F.elu,
                residual=residual,
            )
        )
        self.gat_layers.append(
            GATConv(
                hid_size * heads[0],
                ndim_out,
                heads[1],
                feat_drop=0.,
                attn_drop=0.,
                activation=None,
                residual=residual,
            )
        )
        # self.gat_layers.append(
        #     GATConv(
        #         ndim_in,
        #         ndim_out,
        #         heads[0],
        #         feat_drop=0.,
        #         attn_drop=0.,
        #         activation=F.elu,
        #     )
        # )
        self.mlp = nn.Linear(ndim_out, 2)
        self.res_fc = nn.Linear(ndim_in, ndim_out)

    def forward(self, g, nfeats, corrupt=False, get_embeddings=False):
        res = copy.deepcopy(nfeats)
        h = nfeats
        for i, layer in enumerate(self.gat_layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
            if i == 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        
        if self.residual:
            res = self.res_fc(res)
            h = h + res.squeeze()
        
        if get_embeddings:
            return None, h
    
        return None, self.mlp(h)
