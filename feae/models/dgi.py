import math
from copy import deepcopy

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, hidden_in, hidden_out=None):
      super(Discriminator, self).__init__()

      # ME: GAT need a different hidden_out because of the k heads
      if hidden_out is None:
        hidden_out = hidden_in

      self.weight = nn.Parameter(torch.Tensor(hidden_in, hidden_out))
      self.reset_parameters()

    def uniform(self, size, tensor):
      bound = 1.0 / math.sqrt(size)
      if tensor is not None:
        tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
      size = self.weight.size(0)
      self.uniform(size, self.weight)

    def forward(self, features, summary):
      features = torch.matmul(features, torch.matmul(self.weight, summary))
      return features

class DGI(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, encoder, few_shot_indices, pos_augmentation, neg_augmentation, device, use_hybrid, use_mini_batch=False):
      super(DGI, self).__init__()
      self.encoder = encoder
      self.few_shot_indices = few_shot_indices
      self.device = device
      self.use_hybrid = use_hybrid
      # self.discriminator = Discriminator(128)
      self.discriminator = Discriminator(256)

      self.pos_aug_function = self._add_device_wrapper(pos_augmentation, device)
      self.neg_aug_function = self._add_device_wrapper(neg_augmentation, device)

      self.loss = nn.BCEWithLogitsLoss()
      self.edge_linear = nn.Linear(256, 39)
      # self.edge_linear = nn.Linear(128, 39)

      if use_mini_batch:
        self.forward = self.forward_mini_batch
      else:
        self.forward = self.forward_full_graph

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

    def forward_full_graph(self, g, n_features, e_features):

        # Positive augmentations
        pos_g, pos_n_feats, pos_e_feats = \
            self.pos_aug_function(deepcopy(g), n_features, e_features)
        positive = self.encoder(pos_g, pos_n_feats, pos_e_feats, corrupt=False)
        
        # Negative augmentations
        neg_g, neg_n_feats, neg_e_feats = \
            self.neg_aug_function(deepcopy(g), n_features, e_features)
        negative = self.encoder(neg_g, neg_n_feats, neg_e_feats, corrupt=False)

        positive = positive[1]
        negative = negative[1]

        # DGI stuff
        summary = torch.sigmoid(positive.mean(dim=0))
        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)
        
        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        if self.use_hybrid:

          # Normal reconstruction loss
          recon_feats, target_feats = self.reconstruct_edge_feats_loss_gae(g, n_features, e_features, is_few_shot=False)
          l3 = F.mse_loss(recon_feats, target_feats)

          # Few-shot reconstruction loss
          recon_feats_fs, target_feats_fs = self.reconstruct_edge_feats_loss_gae(g, n_features, e_features, is_few_shot=True)
          l4 = F.mse_loss(recon_feats_fs, target_feats_fs) if len(recon_feats_fs) > 0 else 0.

          return 0.6*l1 + 0.6*l2 - 0.4*l4 + 0.4*l3
        
        return l1 + l2

    def forward_mini_batch(self, g, n_features, e_features):

      # Positive augmentations
      pos_g, pos_n_feats, pos_e_feats = \
          self.pos_aug_function(deepcopy(g), deepcopy(n_features), deepcopy(e_features))
      pos_g.ndata["h"] = pos_n_feats
      pos_g.edata["h"] = pos_e_feats
      # Negative augmentations
      neg_g, neg_n_feats, neg_e_feats = \
          self.neg_aug_function(deepcopy(g), deepcopy(n_features), deepcopy(e_features))
      neg_g.ndata["h"] = neg_n_feats
      neg_g.edata["h"] = neg_e_feats

      sampler = dgl.dataloading.NeighborSampler([20, 20])

      pos_loader = dgl.dataloading.DataLoader(
          pos_g, pos_g.nodes(), sampler,
          batch_size=1024,
          shuffle=False,
          drop_last=False,
          num_workers=0,
          device=self.device,
      )

      neg_loader = dgl.dataloading.DataLoader(
          neg_g, neg_g.nodes(), sampler,
          batch_size=1024,
          shuffle=False,
          drop_last=False,
          num_workers=0,
          device=self.device,
      )

      for (_, _, pos_mfgs), (_, _, neg_mfgs) in zip(pos_loader, neg_loader):

        pos_n_feats = pos_mfgs[0].srcdata['h']
        positive = self.encoder(pos_mfgs, pos_n_feats, pos_mfgs[0].edata["h"], corrupt=False)
        
        neg_n_feats = neg_mfgs[0].srcdata['h']
        negative = self.encoder(neg_mfgs, neg_n_feats, neg_mfgs[0].edata["h"], corrupt=False)

        positive = positive[1]
        negative = negative[1]

        # DGI stuff
        summary = torch.sigmoid(positive.mean(dim=0))
        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)
        
        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        if self.use_hybrid:

          # Normal reconstruction loss
          recon_feats, target_feats = self.reconstruct_edge_feats_loss(g, n_features, e_features, is_few_shot=False)
          l3 = F.mse_loss(recon_feats, target_feats)

          # Few-shot reconstruction loss
          recon_feats_fs, target_feats_fs = self.reconstruct_edge_feats_loss(g, n_features, e_features, is_few_shot=True)
          l4 = F.mse_loss(recon_feats_fs, target_feats_fs)

          return l1 + l2 + l3 - l4
        
        return l1 + l2

    def reconstruct_edge_feats_loss(self, g, n_features, e_features, is_few_shot):
        h_e = self.encoder(g, n_features, e_features, corrupt=False)[1]

        # Removes fex-shot edges
        if is_few_shot:
            mask = torch.zeros(len(h_e), dtype=torch.bool).to(self.device)
            mask[self.few_shot_indices] = True
        else:
            mask = torch.ones(len(h_e), dtype=torch.bool).to(self.device)
            mask[self.few_shot_indices] = False

        h_e = h_e[mask]
        e_features = e_features[mask]

        # h_e = torch.squeeze(h_e)
        # embs = torch.mm(h_e, h_e.T) # (|E|, |E|)
        
        # sinon direct faire un linear sur h_e, 256 => 39
        # if not hasattr(self, "edge_linear"):
        #     self.edge_linear = nn.Linear(embs.shape[-1], e_features.shape[-1])
        
        h_e = torch.sigmoid(self.edge_linear(h_e)) # (|E|, d)
        return h_e, e_features.squeeze()

    def reconstruct_edge_feats_loss_gae(self, g, n_features, e_features, is_few_shot):
        h_e = self.encoder(g, n_features, e_features, corrupt=False)[1]

        # Removes fex-shot edges
        if is_few_shot:
            mask = torch.zeros(len(h_e), dtype=torch.bool).to(self.device)
            mask[self.few_shot_indices] = True
        else:
            mask = torch.ones(len(h_e), dtype=torch.bool).to(self.device)
            mask[self.few_shot_indices] = False

        h_e = h_e[mask]
        e_features = e_features[mask]

        # h_e = torch.squeeze(h_e)
        # embs = torch.mm(h_e, h_e.T) # (|E|, |E|)
        
        # sinon direct faire un linear sur h_e, 256 => 39
        # if not hasattr(self, "edge_linear"):
        #     self.edge_linear = nn.Linear(embs.shape[-1], e_features.shape[-1])
        
        h_e = torch.sigmoid(self.edge_linear(h_e)) # (|E|, d)
        return h_e, e_features.squeeze()
