from copy import deepcopy

import dgl
import torch
import numpy as np
from sklearn.manifold import TSNE

# %%  ========== POSITIVE/NEGATIVE EDGE EMBEDDINGS VIZ ==========
# This time, we want to plot the positive embeddings (the original graph) and the 
# negative embeddings using the same negative sampling technique used during training.
# We should see a binary classification between the 2 classes.

def get_pos_neg_embeddings(ssl_model, pkl_path, pos_aug_function, neg_aug_function, test_g, device, few_shot_indices, other_attack_indices):
    pos_g, pos_n_feats, pos_e_feats = pos_aug_function(deepcopy(test_g.to(device)), test_g.ndata['h'], test_g.edata['h'], device)
    pos_g.ndata['h'] = pos_n_feats
    pos_g.edata['h'] = pos_e_feats

    pos_emb = ssl_model.encoder(pos_g.to(device), pos_g.ndata['h'], pos_g.edata['h'])[1]
    pos_emb = pos_emb.detach().cpu().numpy()

    # Create a negative graph
    neg_g, neg_n_feats, neg_e_feats = neg_aug_function(deepcopy(test_g.to(device)), test_g.ndata['h'], test_g.edata['h'], device)
    neg_g.ndata['h'] = neg_n_feats
    neg_g.edata['h'] = neg_e_feats

    neg_emb = ssl_model.encoder(neg_g.to(device), neg_g.ndata['h'], neg_g.edata['h'])[1]
    neg_emb = neg_emb.detach().cpu().numpy()

    if few_shot_indices != None:
        emb = ssl_model.encoder(test_g.to(device), test_g.ndata['h'], test_g.edata['h'])[1]
        emb = emb.detach().cpu().numpy()

        # Few shot edges
        fs_mask = torch.zeros(len(emb), dtype=torch.bool)
        fs_mask[few_shot_indices] = True

        # All attack edges except few shot ones
        other_attacks_mask = torch.zeros(len(emb), dtype=torch.bool)
        other_attacks_mask[other_attack_indices] = True
        
        # anchor_emb = ssl_model.encoder(test_g.to(device), test_g.ndata['h'], test_g.edata['h'])[1]
        # anchor_emb = pos_emb
        few_shot_emb = emb[fs_mask]
        other_attacks_emb = emb[other_attacks_mask]
        
        return pos_emb, neg_emb, few_shot_emb, other_attacks_emb

    return pos_emb, neg_emb, np.array([]), np.array([])

def get_pos_neg_embeddings_minibatch(ssl_model, pkl_path, pos_aug_function, neg_aug_function, test_g, device):

    pos_g, pos_n_feats, pos_e_feats = pos_aug_function(test_g.to(device), test_g.ndata['h'], test_g.edata['h'], device)
    pos_g.ndata['h'] = pos_n_feats
    pos_g.edata['h'] = pos_e_feats

    sampler = dgl.dataloading.NeighborSampler([20, 20])

    pos_loader = dgl.dataloading.DataLoader(
        pos_g, pos_g.nodes(), sampler,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        device=ssl_model.device,
    )

        # Create a negative graph
    neg_g, neg_n_feats, neg_e_feats = neg_aug_function(test_g.to(device), test_g.ndata['h'], test_g.edata['h'], device)
    neg_g.ndata['h'] = neg_n_feats
    neg_g.edata['h'] = neg_e_feats

    neg_loader = dgl.dataloading.DataLoader(
        neg_g, neg_g.nodes(), sampler,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        device=ssl_model.device,
    )

    (_, _, pos_mfgs), (_, _, neg_mfgs) = next(zip(pos_loader, neg_loader))

    pos_n_feats = pos_mfgs[0].srcdata['h']
    pos_emb = ssl_model.encoder(pos_mfgs, pos_n_feats, pos_mfgs[0].edata["h"], corrupt=False)[1].detach().cpu().numpy()
    
    neg_n_feats = neg_mfgs[0].srcdata['h']
    neg_emb = ssl_model.encoder(neg_mfgs, neg_n_feats, neg_mfgs[0].edata["h"], corrupt=False)[1].detach().cpu().numpy()

    return pos_emb, neg_emb


def tsne(pos_emb, neg_emb, few_shot_emb, other_attacks_emb):
    only_keep = 3000
    print(f"Generating TSNE embeddings for {only_keep} edges...")

    # Add pos and neg embeddings together for plotting
    pos_and_neg_embeds = np.concatenate((pos_emb[: only_keep], neg_emb[: only_keep], other_attacks_emb[: only_keep], few_shot_emb[: only_keep]), axis=0)
    labels = np.concatenate((np.ones(min(only_keep, len(pos_emb))), \
        np.zeros(min(only_keep, len(neg_emb))), \
        np.full((min(only_keep, len(few_shot_emb))), 2), \
        np.full((min(only_keep, len(other_attacks_emb))), 3)), axis=0)

    tsne = TSNE(random_state = 0, n_iter = 1000, metric = 'cosine')
    embeddings2d = tsne.fit_transform(pos_and_neg_embeds)

    num_categories = 4
    xs, ys = [], []

    for label in range(num_categories):
        indices = labels == label
        x = embeddings2d[indices,0]
        y = embeddings2d[indices,1]
        xs.append(x)
        ys.append(y)

    return xs, ys

def get_xs_ys_to_plot(ssl_model, pkl_path, pos_aug_function, neg_aug_function, test_g, device, few_shot_indices, other_attack_indices):
    ssl_model = deepcopy(ssl_model)
    ssl_model.load_state_dict(torch.load(pkl_path, map_location=torch.device('cpu')))
    pos_emb, neg_emb, few_shot_emb, other_attacks_emb = get_pos_neg_embeddings(ssl_model, pkl_path, pos_aug_function, neg_aug_function, test_g, device, few_shot_indices, other_attack_indices)
    xs, ys = tsne(pos_emb, neg_emb, few_shot_emb, other_attacks_emb)
    return xs, ys

# %%  ========== LABEL EMBEDDINGS VIZ ==========
def get_labeled_embeddings(encoder, pkl_path, test_g, test_labels, device):
    emb = encoder(test_g.to(device), test_g.ndata['h'], test_g.edata['h'])[1]
    emb = emb.detach().cpu().numpy()

    return emb

def tsne_labeled(emb, labels):
    only_keep = 3000
    print(f"Generating TSNE embeddings for {only_keep} edges...")

    emb = emb[: only_keep]
    labels = labels[: only_keep]

    tsne = TSNE(random_state = 0, n_iter = 1000, metric = 'cosine')
    embeddings2d = tsne.fit_transform(emb)

    num_categories = 2
    xs, ys = [], []

    for label in range(num_categories):
        indices = (labels == label)
        x = embeddings2d[indices,0]
        y = embeddings2d[indices,1]
        xs.append(x)
        ys.append(y)

    return xs, ys

def get_xs_ys_to_plot_with_labels(encoder, pkl_path, test_g, test_labels, device):
    encoder = deepcopy(encoder)
    encoder.load_state_dict(torch.load(pkl_path, map_location=torch.device('cpu')))
    emb = get_labeled_embeddings(encoder, pkl_path, test_g, test_labels, device)
    xs, ys = tsne_labeled(emb, test_labels)
    return xs, ys