#%%
import argparse
import os
from datetime import datetime
from typing import *

import category_encoders as ce
import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

from augmentations import *
from models import DGI, SAGE, EdgeGCN, GGD, EGAT
from utils import log_regression, ClassificationBestMetrics
from few_shot_utils import get_few_shot_indices

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default="tests")
parser.add_argument('--nb-iter', type=int, default=1)
# parser.add_argument('--augmentation-sample', type=float, default=0.1)

# Dataset args
parser.add_argument('--dataset', type=str, default="all") # ["CSE_CIC", "UNSW", "all"]
parser.add_argument('--dataset-split', type=float, default=0.1)
parser.add_argument('--directed', type=bool, default=False) # default uses the same undirected graph as Anomal-E

# Augmentation args
# ["multi_augmentations", "augment_identity", "augment_perm_all_edges", "augment_drop_edges", "augment_drop_nodes"]
parser.add_argument('--pos-augment', type=str, default="augment_identity")
parser.add_argument('--neg-augment', type=str, default="augment_perm_all_edges")

# Encoder args
parser.add_argument('--encoder', type=str, default="EGCN") # ["EGCN", "EGraphSAGE", "EGAT"]
parser.add_argument('--encoder-patience', type=int, default=250)
parser.add_argument('--encoder-epochs', type=int, default=2000)
parser.add_argument('--egcn-norm', type=str, default="none") # ["left", "both", "right"]
parser.add_argument('--egcn-aggreg', type=str, default="sum") # ["sum", "mean", "max"]
parser.add_argument('--egat-aggreg', type=str, default="mean") # ["sum", "mean", "max"]
parser.add_argument('--egraphsage-aggreg', type=str, default="mean") # ["sum", "mean", "max"]
parser.add_argument('--last-encoder-activation', type=str, default="none") # ["none", "relu", "tanh"]
parser.add_argument('--bias', type=bool, default=False)
parser.add_argument('--residual', type=bool, default=False)

# SSL args
parser.add_argument('--ssl-model', type=str, default="DGI") # ["DGI", "GGD"]
parser.add_argument('--use-hybrid-ssl-loss', type=bool, default=False)
parser.add_argument('--lr', type=float, default=0.001)

# Decoder (few-shot) args
parser.add_argument('--mlp-patience', type=int, default=1500)
parser.add_argument('--mlp-epochs', type=int, default=4000)
parser.add_argument('--mlp-nb-iter', type=int, default=5)
parser.add_argument('--few-shot-sample-size', type=int, default=6) # % of size of the overall few-shot dataset (mal+benign examples)
parser.add_argument('--k', type=int, default=5) # nb of few-shot malicious examples in each attack class

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
# datasets = ["NF-UNSW-NB15-v2.csv"]
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

    data = pd.read_csv("../csnet-contrastive-learning/" + file_name)

    data.rename(columns=lambda x: x.strip(), inplace=True)
    data['IPV4_SRC_ADDR'] = data["IPV4_SRC_ADDR"].apply(str)
    data['L4_SRC_PORT'] = data["L4_SRC_PORT"].apply(str)
    data['IPV4_DST_ADDR'] = data["IPV4_DST_ADDR"].apply(str)
    data['L4_DST_PORT'] = data["L4_DST_PORT"].apply(str)


    #%%
    data.drop(columns=["L4_SRC_PORT", "L4_DST_PORT"], inplace=True)
    data = data.groupby(by='Attack').sample(frac=args.dataset_split if not dataset.startswith("NF-BoT-IoT-v2") else 0.07 , random_state=13)


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
    # train_g = dgl.add_self_loop(train_g)
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
    # test_g = dgl.add_self_loop(test_g)
    nfeat_weight = torch.ones([test_g.number_of_nodes(),
    test_g.edata['h'].shape[1]])
    test_g.ndata['h'] = nfeat_weight

    #%%
    ndim_in = train_g.ndata['h'].shape[-1]
    hidden_features = 128
    ndim_out = 128
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


    #%%
    # Compute the few-shot indices to leverage in the training of the encoder.
    few_shot_indices, few_shot_mal_indices = get_few_shot_indices(
        N=args.k,
        percentage_of_benign=args.few_shot_sample_size / 100,
        train_labels=train_labels,
        train_attack_families=train_attack_families,
    )
    
    fs_mal_indices_set = set(few_shot_mal_indices.tolist())
    other_attack_indices = [i for i, lab in enumerate(train_labels) if lab == 1 and i not in fs_mal_indices_set]

    #%%
    #################### TRAINING ####################

    few_shot_best_metrics = ClassificationBestMetrics()
    supervised_best_metrics = ClassificationBestMetrics()

    for it in range(args.nb_iter):
        log("")
        log(f"=========== START ITERATION {it} =========== ")
        log("")
        log_prefix = f"{args.exp}/{dataset}/it_{it}"
        os.makedirs(log_prefix, exist_ok=True)

        encoder, model_path = None, None

        if args.encoder == "EGAT":
            encoder = EGAT(
                in_size=ndim_in,
                hid_size=hidden_features,
                out_size=hidden_features,
                heads=[4, 2],
                aggreg=args.egat_aggreg,
            )
            model_path = f'{log_prefix}/best_egat.pkl'

        elif args.encoder == "EGCN":
            encoder = EdgeGCN(
                in_feats=ndim_in,
                hidden_feats=ndim_in, # warning: here we need the same shape for broadcasting
                out_feats=ndim_out,
                norm=args.egcn_norm,
                bias=args.bias,
                residual=args.residual,
                aggreg=args.egcn_aggreg,
                last_encoder_activation=args.last_encoder_activation,
            )
            model_path = f'{log_prefix}/best_egcn.pkl'

        elif args.encoder == "EGraphSAGE":
            encoder = SAGE(ndim_in, ndim_out, edim,  F.relu, aggreg=args.egraphsage_aggreg)
            model_path = f'{log_prefix}/best_egraphsage.pkl'

        else:
            raise ValueError("Invalid encoder.")

        # Took the most performant augmentations based on our CSNET paper.
        augments = {
            "multi_augmentations": multi_augmentations([augment_identity, augment_drop_edges]),
            # "multi_augmentations": multi_augmentations([augment_drop_edges, augment_drop_nodes]),
            "augment_identity": augment_identity,
            "augment_perm_all_edges": augment_perm_all_edges,
            "augment_perm_percent_edges": augment_perm_percent_edges,
            "augment_drop_edges": augment_drop_edges,
            "augment_drop_nodes": augment_drop_nodes,
            "augment_mask_all_edges": augment_mask_all_edges,
            "augment_add_edges_rand_features": augment_add_edges_rand_features,
            "augment_mask_percent_edges": augment_mask_percent_edges,
        }

        pos_augmentation = augments[args.pos_augment]
        neg_augmentation = augments[args.neg_augment]

        if args.ssl_model == "DGI":
            dgi = DGI(
                ndim_in,
                ndim_out,
                edim,
                encoder=encoder,
                pos_augmentation=pos_augmentation,
                neg_augmentation=neg_augmentation,
                few_shot_indices=few_shot_mal_indices,
                device=DEVICE,
                use_hybrid=args.use_hybrid_ssl_loss,
            ).to(DEVICE)
        elif args.ssl_model == "GGD":
            dgi = GGD(
                ndim_in,
                ndim_out,
                encoder=encoder,
                pos_augmentation=pos_augmentation,
                neg_augmentation=neg_augmentation,
                few_shot_indices=few_shot_mal_indices,
                device=DEVICE,
                use_hybrid=args.use_hybrid_ssl_loss,
            ).to(DEVICE)
        else:
            raise ValueError("Invalid SSL model.")

        #%%
        cnt_wait = 0
        best = 1e9
        best_t = 0
        node_features = train_g.ndata['h'].to(DEVICE)
        edge_features = train_g.edata['h'].to(DEVICE)
        train_g = train_g.to(DEVICE)

        dgi_optimizer = torch.optim.Adam(
            dgi.parameters(),
            args.lr,
            weight_decay=0.01) # TODO: seems useless

        log(f"Start training for {args.encoder}.")
        for epoch in range(args.encoder_epochs):
            dgi.train()

            dgi_optimizer.zero_grad()
            loss = dgi(train_g, node_features, edge_features)
            loss.backward()
            dgi_optimizer.step()

            # TODO: verify if it's worth it
            torch.nn.utils.clip_grad_norm_(dgi.parameters(), 10)

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(dgi.state_dict(), model_path)
            else:
                cnt_wait += 1

            if cnt_wait == args.encoder_patience:
                log('Early stopping!')
                break

            if epoch == 0 or (epoch+1) % 10 == 0:
                log("Epoch {:04d} | Loss {:.4f} |".format(epoch+1, loss.item()))

        log(f"Best SSL Loss: {best:.4f}")

        #%%
        # dgi.load_state_dict(torch.load(model_path))
        dgi.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        log(f"Model {model_path} loaded.")

        #%%
        test_g = test_g.to(DEVICE)

        #%%
        # Compute train embeddings
        # training_emb = get_embeddings_minibatch(train_g, dgi, DEVICE)
        # testing_emb = get_embeddings_minibatch(test_g, dgi, DEVICE)

        training_emb = dgi.encoder(train_g.to(DEVICE), train_g.ndata['h'].to(DEVICE), train_g.edata['h'].to(DEVICE))[1]
        training_emb = training_emb.detach().cpu().numpy()

        # Compute test embeddings
        testing_emb = dgi.encoder(test_g, test_g.ndata['h'], test_g.edata['h'])[1]
        testing_emb = testing_emb.detach().cpu().numpy()


        #%%
        df_train = pd.DataFrame(training_emb, )
        df_train["Attack"] = train_attack_families
        df_train["Label"] = train_labels

        df_test = pd.DataFrame(testing_emb, )
        df_test["Attack"] = test_attack_families
        df_test["Label"] = test_labels


        #%%
        benign_train_samples = df_train[df_train.Label == 0].drop(columns=["Label", "Attack"])
        normal_train_samples = df_train.drop(columns=["Label", "Attack"])

        test_samples = df_test.drop(columns=["Label", "Attack"])


        #%%
        #################### FEW-SHOT LEARNING ####################
        # For later experiments, we select N samples from each class to have a more balanced set of attacks.
        log(f"Number of edge embeddings: {len(training_emb)}")

        few_shot_emb = training_emb[few_shot_indices]
        few_shot_labels = train_labels[few_shot_indices]

        log(f"Number of malicious samples in few shot training set: {len([i for i in range(len(few_shot_labels)) if few_shot_labels[i] == 1])}")
        log(f"Total number of training samples: {len(few_shot_emb)}")


        #%%
        training_emb = torch.tensor(training_emb).to(DEVICE)
        testing_emb = torch.tensor(testing_emb).to(DEVICE)
        few_shot_emb = torch.tensor(few_shot_emb).to(DEVICE)

        if not isinstance(train_labels, torch.Tensor):
            train_labels = torch.tensor(train_labels)
        if not isinstance(test_labels, torch.Tensor):
            test_labels = torch.tensor(test_labels)
        if not isinstance(few_shot_labels, torch.Tensor):
            few_shot_labels = torch.tensor(few_shot_labels)


        #%%
        # Few shot training on few-shot data
        report = log_regression(few_shot_emb, testing_emb, few_shot_labels, test_labels, \
            log_path=log_prefix, epochs=args.mlp_epochs, patience=args.mlp_patience, mode="few-shot", logger=log, device=device, iterations=args.mlp_nb_iter)
        few_shot_best_metrics.add_report(report)
        log("RESULTS for MLP few-shot:")
        log(report)


    log("")
    log("=========== EXPERIMENTS RESULTS =========== ")
    log("")
    log("")
    log(">>> FEW-SHOT:")
    log("Mean:")
    log(few_shot_best_metrics.get_mean_experiment())
    log("")
    log("Best:")
    log(few_shot_best_metrics.get_best_experiment())

    log("")
    log("")
    log(">>> SUPERVISED:")
    log("Mean:")
    log(supervised_best_metrics.get_mean_experiment())
    log("")
    log("Best:")
    log(supervised_best_metrics.get_best_experiment())
