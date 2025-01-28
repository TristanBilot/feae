import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SGConv
import dgl.function as fn
import numpy as np
from copy import deepcopy


class GGD(nn.Module):
    def __init__(self, in_feats, n_hidden, encoder, device, pos_augmentation, neg_augmentation, few_shot_indices, use_hybrid, proj_layers=1):
        super(GGD, self).__init__()
        self.encoder = encoder
        self.pos_aug_function = self._add_device_wrapper(pos_augmentation, device)
        self.neg_aug_function = self._add_device_wrapper(neg_augmentation, device)
        self.few_shot_indices = few_shot_indices
        self.use_hybrid = use_hybrid
        # self.encoder = Encoder(g, in_feats, n_hidden, n_layers, activation, dropout, gnn_encoder, num_hop)
        self.mlp = torch.nn.ModuleList()
        for i in range(proj_layers):
            self.mlp.append(nn.Linear(256, 256))
        self.loss = nn.BCEWithLogitsLoss()
        # self.graphconv = GraphConv(in_feats, n_hidden, weight=False, bias=False, activation=None)
        self.device = device
        self.edge_linear = nn.Linear(256, 39)

    def _add_device_wrapper(self, augmentation_func, device):
        def wrapper(g, n_feats, e_feats):
            g, n_feats, e_feats = augmentation_func(g, n_feats, e_feats, device)
            n_feats = n_feats.to(device)
            e_feats = e_feats.to(device)
            g = g.to(device)
            g.ndata['h'] = n_feats
            g.edata['h'] = e_feats
            return g, n_feats, e_feats
        return wrapper

    def forward(self, g, node_feats, edge_feats):
        pos_g, pos_n_feats, pos_e_feats = \
            self.pos_aug_function(deepcopy(g), deepcopy(node_feats), deepcopy(edge_feats))
        h_1 = self.encoder(pos_g, pos_n_feats, pos_e_feats, corrupt=False)[1]

        neg_g, neg_n_feats, neg_e_feats = \
            self.neg_aug_function(deepcopy(g), deepcopy(node_feats), deepcopy(edge_feats))
        h_2 = self.encoder(neg_g, neg_n_feats, neg_e_feats, corrupt=True)[1]

        sc_1 = h_1.squeeze(0)
        sc_2 = h_2.squeeze(0)
        for i, lin in enumerate(self.mlp):
            sc_1 = lin(sc_1)
            sc_2 = lin(sc_2)

        sc_1 = sc_1.sum(1).unsqueeze(0)
        sc_2 = sc_2.sum(1).unsqueeze(0)

        lbl_1 = torch.ones(1, sc_1.shape[1])
        lbl_2 = torch.zeros(1, sc_2.shape[1])
        lbl = torch.cat((lbl_1, lbl_2), 1).to(self.device)

        logits = torch.cat((sc_1, sc_2), 1)

        loss = self.loss(logits, lbl)

        if self.use_hybrid:
            # Normal reconstruction loss
            recon_feats, target_feats = self.reconstruct_edge_feats_loss(g, node_feats, edge_feats, is_few_shot=False)
            l3 = F.mse_loss(recon_feats, target_feats)

            # Few-shot reconstruction loss
            recon_feats_fs, target_feats_fs = self.reconstruct_edge_feats_loss(g, node_feats, edge_feats, is_few_shot=True)
            l4 = F.mse_loss(recon_feats_fs, target_feats_fs)

            return loss + l3 - l4

        return loss

    def embed(self, blocks):
        h_1 = self.encoder(blocks, corrupt=False)

        return h_1.detach()

    def reconstruct_edge_feats_loss(self, g, n_features, e_features, is_few_shot):
        h_e = self.encoder(g, n_features, e_features, corrupt=False)[1]

        # Removes fex-shot edges
        if is_few_shot:
            mask = torch.ones(len(h_e), dtype=torch.bool).to(self.device)
            mask[self.few_shot_indices] = False
        else:
            mask = torch.zeros(len(h_e), dtype=torch.bool).to(self.device)
            mask[self.few_shot_indices] = True

        h_e = h_e[mask]
        e_features = e_features[mask]

        h_e = torch.sigmoid(self.edge_linear(h_e)) # (|E|, d)
        return h_e, e_features.squeeze()
