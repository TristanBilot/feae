#%%
import argparse
import os
from datetime import datetime
from typing import *

import category_encoders as ce
import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from matplotlib import colormaps

from augmentations import *
from models import SAGE
from utils import MLP
from viz import get_xs_ys_to_plot_with_labels

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default="tests")
parser.add_argument('--nb-iter', type=int, default=1)

# Dataset args
parser.add_argument('--dataset', type=str, default="all") # ["CSE_CIC", "UNSW", "all"]
parser.add_argument('--dataset-split', type=float, default=0.1)
parser.add_argument('--directed', type=bool, default=True) # For line graphs, undirected cannot fit into memory

# Encoder args
parser.add_argument('--egraphsage-aggreg', type=str, default="mean") # ["sum", "mean", "max"]

parser.add_argument('--encoder', type=str, default="EGraphSAGE") # ["LineGCN", "LineGAT", "LineSAGE"]
parser.add_argument('--encoder-patience', type=int, default=20)
parser.add_argument('--encoder-epochs', type=int, default=600)
parser.add_argument('--egcn-norm', type=str, default="none") # ["left", "both", "right", "none"]
parser.add_argument('--bias', type=bool, default=False)
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--sage-aggreg', type=str, default="mean") # ["gcn", "mean", "max"]
parser.add_argument('--sage-do-sampling', type=bool, default=False)
args = parser.parse_args()


#%%
LOG_FILE = f"{args.exp}/logs.log"

def log(msg, path=LOG_FILE):
  os.makedirs(args.exp, exist_ok=True)
  now = datetime.now()
  formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
  if msg == "": string = ""
  else:
    string = f"{formatted_datetime} : {msg}"

  with open(path, "a+") as f:
    f.write(string + "\n")
  print(string)

log(args)

#%%
if torch.cuda.is_available():
    device = 'cuda'
    log(f'Num GPUs Available: {torch.cuda.device_count()}')
else:
    device = 'cpu'
    log("CPU used.")

DEVICE = torch.device(device)
COMPUTE_DEVICE_CPU = torch.device('cpu')
# DEVICE = COMPUTE_DEVICE_CPU


#%%
datasets = ["NF-CSE-CIC-IDS2018-v2.csv", "NF-UNSW-NB15-v2.csv"]
if args.dataset == "CSE_CIC":
    datasets = [datasets[0]]
elif args.dataset == "UNSW":
    datasets = [datasets[1]]
elif args.dataset == "all":
    pass
else:
    raise ValueError("Invalid dataset.")

#%%
for file_name in datasets:
    dataset = file_name.split('.csv')[0]
    
    log("")
    log(f"=========== START DATASET {dataset} =========== ")
    log("")

    parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = pd.read_csv(os.path.join(parent_folder, file_name))

    data.rename(columns=lambda x: x.strip(), inplace=True)
    data['IPV4_SRC_ADDR'] = data["IPV4_SRC_ADDR"].apply(str)
    data['L4_SRC_PORT'] = data["L4_SRC_PORT"].apply(str)
    data['IPV4_DST_ADDR'] = data["IPV4_DST_ADDR"].apply(str)
    data['L4_DST_PORT'] = data["L4_DST_PORT"].apply(str)


    #%%
    data.drop(columns=["L4_SRC_PORT", "L4_DST_PORT"], inplace=True)
    if dataset.startswith("NF-CSE-CIC-IDS2018-v2-small"):
        data = data.groupby(by='Attack').sample(frac=args.dataset_split, random_state=13)
    else:
        data = data.groupby(by='Attack').sample(frac=args.dataset_split if dataset.startswith("NF-UNSW") else 0.005, random_state=13)


    #%%
    X = data.drop(columns=["Attack", "Label"])
    y = data[["Attack", "Label"]]

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=13, stratify=y)


    #%%
    enc = ce.TargetEncoder(cols=['TCP_FLAGS','L7_PROTO','PROTOCOL',
                                    'CLIENT_TCP_FLAGS','SERVER_TCP_FLAGS','ICMP_TYPE',
                                    'ICMP_IPV4_TYPE','DNS_QUERY_ID','DNS_QUERY_TYPE',
                                    'FTP_COMMAND_RET_CODE'])
    enc.fit(X_train, y_train.Label)

    # Transform on training set
    X_train = enc.transform(X_train)

    # Transform on testing set
    X_test = enc.transform(X_test)


    #%%
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)


    #%%
    scaler = Normalizer()
    cols_to_norm = list(set(list(X_train.iloc[:, 2:].columns))) # Ignore first two as the represents IP addresses
    scaler.fit(X_train[cols_to_norm])

    # Transform on training set
    X_train[cols_to_norm] = scaler.transform(X_train[cols_to_norm])
    X_train['h'] = X_train.iloc[:, 2:].values.tolist()

    # Transform on testing set
    X_test[cols_to_norm] = scaler.transform(X_test[cols_to_norm])
    X_test['h'] = X_test.iloc[:, 2:].values.tolist()

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)


    #%%
    lab_enc = preprocessing.LabelEncoder()
    lab_enc.fit(data["Attack"])

    # Transform on training set
    train["Attack"] = lab_enc.transform(train["Attack"])

    # Transform on testing set
    test["Attack"] = lab_enc.transform(test["Attack"])


    #%%
    # With a real directed graph:
    if args.directed:
        train_g = nx.from_pandas_edgelist(train, "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
                ["h", "Label", "Attack"], create_using=nx.MultiDiGraph())
    else:
        train_g = nx.from_pandas_edgelist(train, "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
                ["h", "Label", "Attack"], create_using=nx.MultiGraph())
        train_g = train_g.to_directed()

    train_g = dgl.from_networkx(train_g, edge_attrs=['h', 'Attack', 'Label'])
    nfeat_weight = torch.ones([train_g.number_of_nodes(),
    train_g.edata['h'].shape[1]])
    train_g.ndata['h'] = nfeat_weight

    # Testing graph
    if args.directed:
        test_g = nx.from_pandas_edgelist(test, "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
                ["h", "Label", "Attack"], create_using=nx.MultiDiGraph())
    else:
        test_g = nx.from_pandas_edgelist(test, "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
                ["h", "Label", "Attack"], create_using=nx.MultiGraph())
        test_g = test_g.to_directed()

    test_g = dgl.from_networkx(test_g, edge_attrs=['h', 'Attack', 'Label'])
    nfeat_weight = torch.ones([test_g.number_of_nodes(),
    test_g.edata['h'].shape[1]])
    test_g.ndata['h'] = nfeat_weight

    #%%
    ndim_in = train_g.ndata['h'].shape[-1]
    hidden_features = 256
    ndim_out = 256
    num_layers = 1
    edim = train_g.edata['h'].shape[-1]


    #%%
    # Format node and edge features for E-GraphSAGE
    train_g.ndata['h'] = torch.reshape(train_g.ndata['h'],
                                    (train_g.ndata['h'].shape[0], 1,
                                        train_g.ndata['h'].shape[1]))

    train_g.edata['h'] = torch.reshape(train_g.edata['h'],
                                    (train_g.edata['h'].shape[0], 1,
                                        train_g.edata['h'].shape[1]))

    # Reshape
    test_g.ndata['h'] = torch.reshape(test_g.ndata['h'],
                                    (test_g.ndata['h'].shape[0], 1,
                                        test_g.ndata['h'].shape[1]))
    test_g.edata['h'] = torch.reshape(test_g.edata['h'],
                                    (test_g.edata['h'].shape[0], 1,
                                        test_g.edata['h'].shape[1]))


    #%%
    train_attack_families = lab_enc.inverse_transform(
            train_g.edata['Attack'].detach().cpu().numpy())
    train_labels = train_g.edata['Label'].detach().cpu().numpy()

    test_attack_families = lab_enc.inverse_transform(
            test_g.edata['Attack'].detach().cpu().numpy())
    test_labels = test_g.edata['Label'].detach().cpu().numpy()


    #%% SUpervised EgraphSAGE
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
        def __init__(self, ndim_in, ndim_out, edim,  activation, aggreg, device):
            super(SAGE, self).__init__()
            self.layers = nn.ModuleList()
            self.layers.append(SAGELayer(ndim_in, edim, 128, F.relu, aggreg=aggreg))
            self.mlp = MLP(ft_in=256, hid=256, nb_classes=2, in_feats=edim).to(device)
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
            return nfeats.sum(1), self.mlp(e_feats.sum(1))



    #%%
    #################### TRAINING ####################
    f1s = []

    for it in range(args.nb_iter):
        log("")
        log(f"=========== START ITERATION {it} =========== ")
        log("")
        log_prefix = f"{args.exp}/{dataset}/it_{it}"
        os.makedirs(log_prefix, exist_ok=True)

        encoder = SAGE(ndim_in, ndim_out, edim,  F.relu, aggreg=args.egraphsage_aggreg,  device=device)
        model_path = f'{log_prefix}/best_egraphsage.pkl'


        #%%
        cnt_wait = 0
        best, best_f1 = 1e9, -1
        best_t = 0
        node_features = train_g.ndata['h'].to(DEVICE)
        edge_features = train_g.edata['h'].to(DEVICE)

        test_nfeats = test_g.ndata["h"].to(DEVICE)
        test_efeats = test_g.edata['h'].to(DEVICE)
        
        train_g = train_g.to(DEVICE)
        test_g = test_g.to(DEVICE)
        encoder = encoder.to(DEVICE)
        train_lbls = torch.tensor(train_labels, device=DEVICE)

        loss_fn = F.cross_entropy

        optimizer = torch.optim.Adam(
            encoder.parameters(),
            args.lr,
            weight_decay=0.)

        log(f"Start training for {args.encoder}.")
        for epoch in range(args.encoder_epochs):
            encoder.train()

            optimizer.zero_grad()
            preds = encoder(train_g, node_features, edge_features)[1]
            loss = loss_fn(preds, train_lbls)
            loss.backward()
            optimizer.step()

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(encoder.state_dict(), model_path)
            else:
                cnt_wait += 1

            if cnt_wait == args.encoder_patience:
                log('Early stopping!')
                break

            if epoch == 0 or (epoch+1) % 1 == 0:
                encoder.eval()

                test_preds = encoder(test_g, test_nfeats, test_efeats)[1]
                test_preds = torch.argmax(test_preds, dim=1)
                f1 = f1_score(test_labels, test_preds.detach().cpu(), average="macro")

                if f1 > best_f1:
                    best_f1 = f1
                    log("Epoch {:04d} | Loss {:.4f} | Best F1 {:.4f} *".format(epoch+1, loss.item(), best_f1))
                else:
                    if epoch == 0 or (epoch+1) % 10 == 0:
                        log("Epoch {:04d} | Loss {:.4f} | F1 {:.4f}".format(epoch+1, loss.item(), f1))

        log(f"Best Supervised Loss: {best:.4f}")
        log(f"Best F1: {best_f1:.4f}")
        f1s.append(best_f1)

        #%%
        # encoder.load_state_dict(torch.load(model_path))
        encoder.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        log(f"Model {model_path} loaded.")

        #%% VIZ
        model_to_viz = model_path

        cmap = colormaps['tab20']
        fig, ax = plt.subplots(figsize=(8,8))
        xs, ys = get_xs_ys_to_plot_with_labels(encoder, model_to_viz, test_g, test_labels, DEVICE)

        annotations = ["benign", "attack"]
        for label, (x, y) in enumerate(zip(xs, ys)):
            ax.scatter(x, y, c=np.array(cmap(label)).reshape(1,4), label=annotations[label], alpha=0.5)
        ax.legend(fontsize='large', markerscale=2)

        svg_path = f'{log_prefix}/classif_embeds.svg'
        plt.savefig(svg_path)
        log(f"SVG saved to {svg_path}.")

        #%%
        log("")
        log(f"Metrics of {dataset} exp:")
        log(f"Mean F1: {np.mean(f1s).round(4)}")
        log(f"Std F1: {np.std(f1s).round(4)}")
        log(f"Max F1: {np.max(f1s).round(4)}")
