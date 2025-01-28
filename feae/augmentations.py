from random import randrange

import numpy as np
import torch
import dgl

PERCENT_AUGMENT = 0.3

# ---- IDENTITY AUGMENTATION ----
def augment_identity(g, n_feats, e_feats, device):
    """
    Edge case for the augmentations' benchmark. This function actually just returns the graph as is.
    """
    return g, n_feats, e_feats


# ---- ADDING AUGMENTATIONS ----
def augment_add_edges_rand_features(g, n_feats, e_feats, device):
    """
    Adds a percent of new edges with random features ranging from 0 to 1.
    """
    # Add edge features
    nb_edges_to_add = int(len(e_feats) * PERCENT_AUGMENT)
    feature_shape = g.ndata['h'].shape[-1]
    new_e_feats = torch.tensor(np.random.rand(nb_edges_to_add, 1, feature_shape), dtype=torch.float32, device=device)
    e_features = torch.cat([e_feats, new_e_feats], dim=0)
    
    # Add edges
    max_node = g.nodes().max().item()
    rand_src = np.random.randint(0, max_node, size=nb_edges_to_add)
    rand_dst = np.random.randint(0, max_node, size=nb_edges_to_add)

    g.add_edges(u=rand_src, v=rand_dst)

    return g, n_feats, e_features

def augment_add_edges_real_features(g, n_feats, e_feats, device):
    """
    Adds a percent of new edges with features uniformly peaked from other edges in the graph.
    """
    # Add edge features
    nb_edges_to_add = int(len(e_feats) * PERCENT_AUGMENT)

    # Peak some indices to take the features
    indices = np.random.choice(g.number_of_edges(), nb_edges_to_add, replace=False)
    new_e_feats = e_feats[indices]
    e_features = torch.cat([e_feats, new_e_feats], dim=0)
    
    # Add edges
    max_node = g.nodes().max().item()
    rand_src = np.random.randint(0, max_node, size=nb_edges_to_add)
    rand_dst = np.random.randint(0, max_node, size=nb_edges_to_add)

    g_copy = g
    g_copy.add_edges(u=rand_src, v=rand_dst)

    return g_copy, n_feats, e_features

def augment_add_nodes_real_features(g, n_feats, e_feats, device):
    """
    Adds a percent of new nodes and incoming/outgoing edges with features uniformly peaked from other edges in the graph.
    """
    nb_nodes_to_add = int(g.number_of_nodes() * PERCENT_AUGMENT)

    last_node = g.number_of_nodes()
    last_added_node = last_node + nb_nodes_to_add

    nb_incoming_edges = int(g.in_degrees().float().mean())
    nb_outgoing_edges = int(g.out_degrees().float().mean())

    # Add n nodes
    g.add_nodes(nb_nodes_to_add)

    # Create the edges
    src_edges, dst_edges = [], []
    for new_node in range(last_node, last_added_node):

        # Outgoing edges
        rand_dst = np.random.choice(g.number_of_nodes(), nb_outgoing_edges, replace=True)
        src = [new_node] * nb_outgoing_edges

        # Incoming edges
        rand_src = np.random.choice(g.number_of_nodes(), nb_incoming_edges, replace=True)
        dst = [new_node] * nb_incoming_edges

        src_edges.extend([*src, *rand_src])
        dst_edges.extend([*rand_dst, *dst])

    g.add_edges(u=src_edges, v=dst_edges)

    # Append the edge features
    nb_edges_to_add = len(src_edges)
    indices = np.random.choice(g.number_of_edges(), nb_edges_to_add, replace=False)
    new_e_feats = e_feats[indices]
    e_features = torch.cat([e_feats, new_e_feats], dim=0)

    # Append the node features
    n_feats = torch.cat([n_feats, torch.ones(nb_nodes_to_add, *n_feats.shape[1:])], dim=0)

    return g, n_feats, e_features


# ---- DELETION AUGMENTATIONS ----
def augment_drop_edges(g, n_feats, e_feats, device):
    """
    Removes a percentage of edges in the graph.
    """
    nb_edges_to_keep = int(len(e_feats) * (1 - PERCENT_AUGMENT))

    # Generate random indices to remove
    indices = torch.randperm(len(e_feats))[:nb_edges_to_keep].to(device)

    # Create a new tensor without the selected indices
    e_features = e_feats[indices, :]

    # Get only corresponding edges
    subgraph = dgl.edge_subgraph(g, indices)

    # Get only good number of node feats
    n_feats = torch.ones(subgraph.number_of_nodes(), *n_feats.shape[1:]).to(device)

    return subgraph, n_feats, e_features

def augment_drop_nodes(g, n_feats, e_feats, device):
    """
    Removes a percentage of nodes in the graph.
    """
    nb_nodes_to_rm = int(len(n_feats) * PERCENT_AUGMENT)

    # Generate random indices to remove
    indices = torch.randperm(len(n_feats))[:nb_nodes_to_rm].to(device)

    # Just get the subgraph with correponding features
    subgraph = dgl.remove_nodes(g, indices)
    n_feats = subgraph.ndata["h"]
    e_feats = subgraph.edata["h"]
    
    return subgraph, n_feats, e_feats


# ---- PERMUTATION AUGMENTATIONS ----
def augment_perm_all_edges(g, n_feats, e_feats, device):
    """
    Permutes all edges in the graph.
    """
    e_perm = torch.randperm(g.number_of_edges())
    e_feats = e_feats[e_perm]
    
    return g, n_feats, e_feats

def augment_perm_percent_edges(g, n_feats, e_feats, device):
    """
    Permutes only a percentage of edges in the graph.
    """
    nb_edges_to_perm = int(len(e_feats) * PERCENT_AUGMENT)

    # Generate random indices to permute
    indices = np.random.choice(g.number_of_edges(), nb_edges_to_perm, replace=False)

    # Permute the selected elements
    permuted_array = torch.tensor(np.random.permutation(e_feats[indices].cpu()), device=device)

    # Create a copy of the original array
    permuted_edges = e_feats

    # Update the selected elements with the permuted values
    permuted_edges[indices] = permuted_array

    return g, n_feats, permuted_edges


# ---- MASKING AUGMENTATIONS ----
def augment_mask_percent_edges(g, n_feats, e_feats, device):
    """
    Mask a percentage of edges using a gaussian noise.
    """
    nb_edges_to_mask = int(len(e_feats) * PERCENT_AUGMENT)

    # The mask gets only a percentage of edges, and we permut with the edge features
    mask = np.random.choice(len(e_feats), nb_edges_to_mask, replace=False)
    e_feats[mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(nb_edges_to_mask, *e_feats.shape[1:])), dtype=torch.float32, device=device)

    return g, n_feats, e_feats

def augment_mask_all_edges(g, n_feats, e_feats, device):
    """
    Mask all edges with a gaussian distribution.
    """
    e_feats = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(e_feats.shape)), dtype=torch.float32, device=device)

    return g, n_feats, e_feats

# ---- MULTI AUGMENTATIONS ----
def multi_augmentations(augments):
    """
    Randomly peeks one among the given `augments` and returns the result.
    """
    def wrapper(g, n_feats, e_feats, device):
        rand = randrange(len(augments))
        augment = augments[rand]

        return augment(g, n_feats, e_feats, device)

    return wrapper
