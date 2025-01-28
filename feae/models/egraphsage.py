import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation, aggreg):
      super(SAGELayer, self).__init__()
      self.W_apply = nn.Linear(ndim_in + edims , ndim_out)
      self.activation = F.relu
      self.W_edge = nn.Linear(128 * 2, 256)
      self.reset_parameters()

      if aggreg == "sum":
        self.aggreg = fn.sum('m', 'h_neigh')
      elif aggreg == "mean":
        self.aggreg = fn.mean('m', 'h_neigh')
      elif aggreg == "max":
        self.aggreg = fn.max('m', 'h_neigh')
      else:
        raise ValueError("Invalid aggregation function.")

    def reset_parameters(self):
      gain = nn.init.calculate_gain('relu')
      nn.init.xavier_uniform_(self.W_apply.weight, gain=gain)

    def message_func(self, edges):
      return {'m':  edges.data['h']}

    def forward(self, g_dgl, nfeats, efeats):
      with g_dgl.local_scope():
        g = g_dgl
        g.ndata['h'] = nfeats
        g.edata['h'] = efeats
        g.update_all(self.message_func, self.aggreg)
        g.ndata['h'] = F.relu(self.W_apply(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))

        # Compute edge embeddings
        u, v = g.edges()
        edge = self.W_edge(torch.cat((g.srcdata['h'][u], g.dstdata['h'][v]), 2))
        return g.ndata['h'], edge


#%%
class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim,  activation, aggreg):
      super(SAGE, self).__init__()
      self.layers = nn.ModuleList()
      self.layers.append(SAGELayer(ndim_in, edim, 128, F.relu, aggreg=aggreg))
      # self.layers.append(SAGELayer(128, edim, 128, F.relu, aggreg=aggreg))

    def forward(self, g, nfeats, efeats, corrupt=False, do_sample=False):
      if do_sample:
        g = dgl.sampling.sample_neighbors(g, g.nodes(), 20)
      if corrupt:
        e_perm = torch.randperm(g.number_of_edges())
        #n_perm = torch.randperm(g.number_of_nodes())
        efeats = g.edata["h"][e_perm]
        #nfeats = nfeats[n_perm]
      for i, layer in enumerate(self.layers):
        #nfeats = layer(g, nfeats, g.edata["h"])
        nfeats, e_feats = layer(g, g.ndata["h"], g.edata["h"])
      #return nfeats.sum(1)
      return nfeats.sum(1), e_feats.sum(1)
