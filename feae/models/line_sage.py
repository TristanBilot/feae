import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv

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

    node_features = mfgs[0].srcdata['h']
    return mfgs, node_features

class LineSAGE(nn.Module):
    def __init__(self, ndim_in, hid_size, ndim_out, residual, norm, aggreg="mean", neigh_sampling=True):
        super(LineSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.residual = residual
        # two-layer SAGE
        self.conv1 = SAGEConv(ndim_in, hid_size, aggregator_type=aggreg)
        self.conv2 = SAGEConv(hid_size, ndim_out, aggregator_type=aggreg)
        self.mlp = nn.Linear(hid_size, 2)

        self.dropout = nn.Dropout(0.)
        self.res_fc = nn.Linear(ndim_in, ndim_out, bias=False)

        if neigh_sampling:
            self.forward = self.forward_sample_neigh
        else:
            self.forward = self.forward_full_neigh

    def forward_full_neigh(self, g, node_feats, corrupt=False, get_embeddings=False):
        h = node_feats
        
        h = self.conv1(g, h)
        h = F.relu(h)

        h = self.dropout(h)
        h = self.conv2(g, h)
        
        h = h.squeeze()
        if self.residual:
            res = self.res_fc(node_feats)
            h = h + res

        if get_embeddings:
            return None, h
        return None, self.mlp(h)


    def forward_sample_neigh(self, g, nfeats, corrupt=False, get_embeddings=False):
        g.ndata["h"] = nfeats
        mfgs, nfeats = sample(g)
        h = nfeats
        
        h_dst = h[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (h, h_dst))

        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
            
        h = h.squeeze()
        if self.residual:
            res = self.res_fc(nfeats)
            h = h + res

        if get_embeddings:
            return None, h
        return None, self.mlp(h)
