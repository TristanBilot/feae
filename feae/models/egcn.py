import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.utils import expand_as_pair


def sample(g):
    sampler = dgl.dataloading.NeighborSampler([20, 20])
    train_dataloader = dgl.dataloading.DataLoader(
        g,              # The graph
        g.nodes(),         # The node IDs to iterate over in minibatches
        sampler,            # The neighbor sampler
        device=g.device,      # Put the sampled MFGs on CPU or GPU
        batch_size=100000000,    # No batching
    )
    input_nodes, output_nodes, mfgs = next(iter(train_dataloader))

    nfeatures = mfgs[0].srcdata['h']
    return mfgs, nfeatures

class EdgeGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, norm, bias, aggreg):
        super(EdgeGCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self._norm = norm
        
        # if bias:
        #     self._bias = nn.Parameter(torch.Tensor(out_feats))

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
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)

    def forward(self, g, node_feats, edge_feats):
        """Inspired from https://docs.dgl.ai/en/1.1.x/_modules/dgl/nn/pytorch/conv/graphconv.html#GraphConv.forward
        """
        with g.local_scope():
            g.ndata['h'] = node_feats
            g.edata['h'] = edge_feats

            feat_src, feat_dst = expand_as_pair(node_feats, g)

            if self._norm in ["left", "both"]:
                degs = g.out_degrees().to(feat_src).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm
                g.srcdata["h"] = feat_src
                
            def message_func(edges):
                return {'m':  edges.data['h']}
            g.update_all(message_func, self.aggreg)
            
            # dot product
            rst = g.dstdata["h_neigh"]
            rst = self.linear(rst)

            # right norm
            if self._norm in ["right", "both"]:
                degs = g.in_degrees().to(feat_dst).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            # if hasattr(self, "_bias"):
            #     rst = rst + self._bias

            return rst


class EdgeGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, norm, bias, residual, last_encoder_activation, aggreg, sample_neigh=False):
        super(EdgeGCN, self).__init__()
        self.gcn1 = EdgeGCNLayer(in_feats, out_feats, norm=norm, bias=bias, aggreg=aggreg)
        self.W_edge = nn.Linear(out_feats * 2, 256)
        self.residual = residual
        self.last_encoder_activation = last_encoder_activation
        self.residual_projection = nn.Linear(in_feats, out_feats*2)

        if sample_neigh:
            self.forward = self.forward_sample_neigh
        else:
            self.forward = self.forward_full_neigh

    def forward_full_neigh(self, g, node_feats, edge_feats, corrupt=False):

        # ME: Inspired by the AnomalE SAGE forward corruption method
        if corrupt:
            e_perm = torch.randperm(g.number_of_edges())
            edge_feats = edge_feats[e_perm]

        # add drop edge
        h = self.gcn1(g, node_feats, edge_feats)
        h = F.relu(h)
        # add dropout
        # h = self.gcn2(g, h, edge_feats)

        h_saved = g.ndata["h"]
        g.ndata["h"] = h

        u, v = g.edges()
        edge_embs = torch.cat((g.srcdata['h'][u], g.dstdata['h'][v]), 2)
        if self.residual:
            # target_shape = (1, 256)
            # padding_length = target_shape[-1] - edge_feats.shape[-1]
            # padded_edge_feats = F.pad(edge_feats, (0, padding_length), mode='constant', value=0)
            edge_feats = self.residual_projection(edge_feats)
            edge_embs = edge_embs + edge_feats

        edge_embs = self.W_edge(edge_embs)
        edge_embs = torch.squeeze(edge_embs)

        g.ndata["h"] = h_saved

        # edge_embs = torch.cat([torch.cat([h[u], h[v]], dim=1) for u, v in zip(g.edges()[0], g.edges()[1])])
        return h, edge_embs

    def forward_sample_neigh(self, g, nfeats, efeats, corrupt=False):
        g.ndata["h"] = nfeats
        g.edata["h"] = efeats
        # mfgs, nfeats = sample(g)
        # h = nfeats

        g = dgl.sampling.sample_neighbors(g, g.nodes(), 50)
        h = self.gcn1(g, g.ndata["h"], g.edata["h"])
        h = F.relu(h)
        
        # h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.gcn2(g, h, g.edata["h"])

        h_saved = g.ndata["h"]
        g.ndata["h"] = h

        u, v = g.edges()
        edge_embs = torch.cat((g.srcdata['h'][u], g.dstdata['h'][v]), 2)

        edge_embs = self.W_edge(edge_embs)
        edge_embs = torch.squeeze(edge_embs)

        if self.residual:
            edge_feats = self.residual_projection(g.edata["h"]).squeeze()
            edge_embs = edge_embs + edge_feats

        g.ndata["h"] = h_saved
        return h, edge_embs
