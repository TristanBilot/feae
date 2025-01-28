"""Torch modules for graph attention networks with fully valuable edges (EGAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import dgl.function as fn


class EGAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads, aggreg, use_mini_batch=False):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # two-layer GAT
        self.gat_layers.append(
            EGATConv(in_size, in_size, hid_size, hid_size, heads[0], activation=F.elu, first_layer=True, aggreg=aggreg)
        )
        self.gat_layers.append(
            EGATConv(
                hid_size * heads[0],
                hid_size * heads[0] if not use_mini_batch else in_size,
                out_size,
                out_size,
                heads[1],
                # residual=True,
                activation=None,
                first_layer=False,
                aggreg=aggreg,
            )
        )
        if use_mini_batch:
            self.forward = self.forward_mini_batch
        else:
            self.forward = self.forward_full_graph

    def forward_full_graph(self, g, nfeats, efeats, corrupt=False):
        h_n = nfeats
        h_e = efeats
        for i, layer in enumerate(self.gat_layers):
            h_n, h_e = layer(g, h_n, h_e)
            if i == len(self.gat_layers)-1:  # last layer
                h_n = h_n.mean(1)
                h_e = h_e.mean(1)
            else:  # other layer(s)
                h_n = h_n.flatten(1)
                h_e = h_e.flatten(1)

        return h_n, h_e


    def forward_mini_batch(self, mfgs, nfeats, efeats, corrupt=False):
        h_src = nfeats
        h_e = efeats

        for i, layer in enumerate(self.gat_layers):
            h_dst = h_src[:mfgs[i].num_dst_nodes()]
            h_src, h_e = layer(mfgs[i], (h_src, h_dst), mfgs[i].edata["h"])
            
            # To squeeze the nb head dimension
            if i == len(self.gat_layers)-1:  # last layer
                h_src = h_src.mean(1)
                h_e = h_e.mean(1)
            else:  # other layer(s)
                h_src = h_src.flatten(1)
                h_e = h_e.flatten(1)

        return h_src, h_e

# pylint: enable=W0235
class EGATConv(nn.Module):
    def __init__(self,
                 in_node_feats,
                 in_edge_feats,
                 out_node_feats,
                 out_edge_feats,
                 num_heads,
                 aggreg,
                 first_layer,
                 activation,
                 **kw_args):
        
        super().__init__()
        self.num_heads = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.fc_nodes = nn.Linear(in_node_feats, out_node_feats*num_heads, bias=True)
        self.fc_edges = nn.Linear(in_edge_feats + 2*in_node_feats, out_edge_feats*num_heads, bias=False)
        self.fc_attn = nn.Linear(out_edge_feats, num_heads, bias=False)
        self.reset_parameters()
        self.first_layer = first_layer
        self.activation = activation

        if aggreg == "sum":
            self.aggreg = fn.sum('m', 'h')
        elif aggreg == "mean":
            self.aggreg = fn.mean('m', 'h')
        elif aggreg == "max":
            self.aggreg = fn.max('m', 'h')
        else:
            raise ValueError("Invalid aggregation function.")

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        gain = init.calculate_gain('relu')
        init.xavier_normal_(self.fc_nodes.weight, gain=gain)
        init.xavier_normal_(self.fc_edges.weight, gain=gain)
        init.xavier_normal_(self.fc_attn.weight, gain=gain)

    def edge_attention(self, edges):
        #extract features
        h_src = edges.src['h'].squeeze()
        h_dst = edges.dst['h'].squeeze()
        f = edges.data['f'].squeeze()
        #stack h_i | f_ij | h_j
        stack = th.cat([h_src, f, h_dst], dim=-1)
        # apply FC and activation
        f_out = self.fc_edges(stack)
        f_out = nn.functional.leaky_relu(f_out)
        f_out = f_out.view(-1, self.num_heads, self._out_edge_feats)
        # apply FC to reduce edge_feats to scalar
        a = self.fc_attn(f_out).sum(-1).unsqueeze(-1)

        return {'a': a, 'f' : f_out}

    def message_func(self, edges):
        return {'h': edges.src['h'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        alpha = nn.functional.softmax(nodes.mailbox['a'], dim=1)
        h = th.sum(alpha * nodes.mailbox['h'], dim=1)
        return {'h': h}

    def forward(self, g, nfeats, efeats, corrupt=False):
        with g.local_scope():
            if isinstance(nfeats, tuple):
                h_src, h_dst = nfeats
                g.srcdata['h'] = h_src
                g.dstdata['h'] = h_dst
            else:
                g.ndata['h'] = nfeats
            
            g.edata['f'] = efeats

            if self.first_layer:
                def mean_edge_feats(edges):
                    return {'m':  edges.data['h']}
                g.update_all(mean_edge_feats, self.aggreg)

            g.apply_edges(self.edge_attention)

            nfeats_ = self.fc_nodes(g.srcdata['h'])
            nfeats_ = nfeats_.view(-1, self.num_heads, self._out_node_feats)

            g.srcdata['h'] = nfeats_
            g.update_all(message_func = self.message_func,
                         reduce_func = self.reduce_func)

            # ME: to match the same output shape format as in SAGE, we want a shape (N*d)
            # but initially we have (N*k*d) where k is the number of heads. So we concatenate to obtain (N*k.d)
            node_embs = g.dstdata.pop('h')
            edge_embs = g.edata.pop('f')

            if self.activation:
                node_embs = self.activation(node_embs)
                edge_embs = self.activation(edge_embs)

            return node_embs, edge_embs