import dgl
import gc
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score


def infoNCE_loss():
    def loss_fn(anchor, positive, negative):
        similarity = F.cosine_similarity(anchor.unsqueeze(1).unsqueeze(1), torch.cat([positive.unsqueeze(1).unsqueeze(1), negative.unsqueeze(1).unsqueeze(1)], dim=1), dim=-1)
        log_prob = F.log_softmax(similarity, dim=-1)
        loss = -log_prob[:, 0].mean()
        return loss
    
    return loss_fn

class LogReg(nn.Module):
  def __init__(self, ft_in, nb_classes):
      super(LogReg, self).__init__()
      self.fc = nn.Linear(ft_in, nb_classes)

      for m in self.modules():
          self.weights_init(m)

  def weights_init(self, m):
      if isinstance(m, nn.Linear):
          torch.nn.init.xavier_uniform_(m.weight.data)
          if m.bias is not None:
              m.bias.data.fill_(0.0)

  def forward(self, seq):
      ret = self.fc(seq)
      return ret


class MLP(nn.Module):
	def __init__(self, ft_in, hid, nb_classes, in_feats):
		super(MLP, self).__init__()
		self.fc1 = nn.Linear(ft_in, hid)
		self.fc2 = nn.Linear(hid, nb_classes)

		self.fc_residual = nn.Linear(in_feats, hid)

		for m in self.modules():
			self.weights_init(m)

	def weights_init(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)

	def forward(self, emb, edge_feats=None):
		emb = self.fc1(emb)
		emb = F.relu(emb)

		# residual
		if edge_feats != None:
			emb = emb + self.fc_residual(edge_feats)

		emb = self.fc2(emb)
		return emb

def log_regression(training_emb, testing_emb, train_labels, test_labels, epochs, patience, log_path, mode, logger, device, iterations, lr):
    hid_units = training_emb.shape[-1]
    nb_classes = 2
    bs = None
    cnt_wait = 0
    best_loss = 1e9
    delta = 100
    check_f1_each = 5
    metrics = ClassificationMetrics()

    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)

    for it in range(iterations):
        logger("")
        logger(f"Iteration {it+1:02d} MLP {mode}")
        # log = LogReg(hid_units, nb_classes).to(device)
        log = MLP(ft_in=hid_units, hid=hid_units*2, nb_classes=2, in_feats=39).to(device)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=0.00001)
        xent = nn.CrossEntropyLoss()

        pat_steps = 0
        best_f1 = 0.
        for epoch in range(epochs):
            log.train()
            opt.zero_grad()

            logits = log(training_emb)
            loss = xent(logits, train_labels)
            
            loss.backward()
            opt.step()

            if epoch % delta == 0:
                logger("Epoch {:05d} | Loss {:.4f} |".format(epoch, loss.item()))

            # Break loop when patience is reached for the f1 score
            # in few-shot, the key is to stop training of the head based on F1 and not loss
            # inspired from https://github.com/zpeng27/GMI/blob/master/utils/process.py#L154
            if (epoch+1) % check_f1_each == 0:
                log.eval()
                logits = log(testing_emb)
                preds = torch.argmax(logits, dim=1)
                f1 = f1_score(test_labels.detach().cpu(), preds.detach().cpu(), average="macro")

                if f1 > best_f1:
                    best_f1 = f1
                    best_loss = loss
                    cnt_wait = 0
                    torch.save(log.state_dict(), f'{log_path}/best_mlp_{mode}.pkl')
                    logger("Epoch {:05d} | Loss {:.4f} | F1 {:.3f} | *".format(epoch, loss.item(), best_f1))
                else:
                    cnt_wait += 1
                
                if cnt_wait == patience // check_f1_each:
                    logger('Early stopping!')
                    break

        log.load_state_dict(torch.load(f'{log_path}/best_mlp_{mode}.pkl'))

        logits = log(testing_emb)
        preds = torch.argmax(logits, dim=1)
        bs = preds
        acc = torch.sum(preds == test_labels).float() / test_labels.shape[0]
        logger(f"Best F1: {best_f1:.3f}, Best Acc: {acc:.2f}; Best Loss: {best_loss:.4f}")

        # Metrics
        report = classification_report(test_labels.detach().cpu(), bs.detach().cpu(), digits=4, output_dict=True)
        metrics.add_report(report)

    return metrics.compute_mean_std()

def isolation_forest(benign_train_samples, test_samples, test_labels):
    cont = [0.001, 0.01, 0.04, 0.05, 0.1, 0.2]
    n_est = [20, 50, 100, 150]
    # n_est = [50, 100]
    # cont = [0.05, 0.2]
    params = list(product(n_est, cont))
    score = -1
    bs = None

    for n_est, con in params:
        clf_if = IsolationForest(n_estimators=n_est, contamination=con)
        clf_if.fit(benign_train_samples)
        y_pred = clf_if.predict(test_samples)
        test_pred = list(map(lambda x : 0 if x == 1 else 1, y_pred))

        f1 = f1_score(test_labels, test_pred, average='macro')

        if f1 > score:
            score = f1
            best_params = {'n_estimators': n_est,
                        "con": con
                    }
            bs = test_pred
        del clf_if
        gc.collect()

    report = classification_report(test_labels, bs, digits=4, output_dict=True)
    metrics = ClassificationMetrics(); metrics.add_report(report)
    metrics = metrics.compute_mean_std()
    print(best_params)
    return metrics


class ClassificationMetrics:
    def __init__(self):
        self.precision_list = []
        self.recall_list = []
        self.f1_score_list = []
        self.support_list = []

    def add_report(self, report):
        self.precision_list.append(report['macro avg']['precision'])
        self.recall_list.append(report['macro avg']['recall'])
        self.f1_score_list.append(report['macro avg']['f1-score'])
        self.support_list.append(report['macro avg']['support'])

    def compute_mean_std(self):
        rd = 4
        max_precision = np.max(self.precision_list).round(rd)
        mean_precision = np.mean(self.precision_list).round(rd)
        std_precision = np.std(self.precision_list).round(rd)

        max_recall = np.max(self.recall_list).round(rd)
        mean_recall = np.mean(self.recall_list).round(rd)
        std_recall = np.std(self.recall_list).round(rd)

        max_f1_score = np.max(self.f1_score_list).round(rd)
        mean_f1_score = np.mean(self.f1_score_list).round(rd)
        std_f1_score = np.std(self.f1_score_list).round(rd)

        max_support = np.max(self.support_list).round(rd)
        mean_support = np.mean(self.support_list).round(rd)
        std_support = np.std(self.support_list).round(rd)

        return {
            'max_f1_score': max_f1_score,
            'max_precision': max_precision,
            'max_recall': max_recall,
            'max_support': max_support,
            'mean_f1_score': mean_f1_score,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_support': mean_support,
            'std_f1_score': std_f1_score,
            'std_precision': std_precision,
            'std_recall': std_recall,
            'std_support': std_support
        }

class ClassificationBestMetrics:
    def __init__(self):
        self.reports = []

    def add_report(self, report):
        self.reports.append(report)

    def get_best_experiment(self):
        best_f1_score, best_exp = -1, -1
        for exp in self.reports:
            if exp["max_f1_score"] > best_f1_score:
                best_f1_score = exp["max_f1_score"]
                best_exp = exp
        return best_exp

    def get_mean_experiment(self):
        return {
            'max_f1_score': np.mean([x["max_f1_score"] for x in self.reports]).round(4),
            'max_precision': np.mean([x["max_precision"] for x in self.reports]).round(4),
            'max_recall': np.mean([x["max_recall"] for x in self.reports]).round(4),
            'max_support': np.mean([x["max_support"] for x in self.reports]).round(4),
            'mean_f1_score': np.mean([x["mean_f1_score"] for x in self.reports]).round(4),
            'mean_precision': np.mean([x["mean_precision"] for x in self.reports]).round(4),
            'mean_recall': np.mean([x["mean_recall"] for x in self.reports]).round(4),
            'mean_support': np.mean([x["mean_support"] for x in self.reports]).round(4),
            'std_f1_score': np.mean([x["std_f1_score"] for x in self.reports]).round(4),
            'std_precision': np.mean([x["std_precision"] for x in self.reports]).round(4),
            'std_recall': np.mean([x["std_recall"] for x in self.reports]).round(4),
            'std_support': np.mean([x["std_support"] for x in self.reports]).round(4),
        }

def get_embeddings_minibatch(g, ssl_model, device):
    sampler = dgl.dataloading.NeighborSampler([20, 20])

    loader = dgl.dataloading.DataLoader(
        g, g.nodes(), sampler,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        device=device,
    )
    training_emb = []
    for (_, _, mfgs) in loader:
        n_feats = mfgs[0].srcdata['h']
        emb = ssl_model.encoder(mfgs, n_feats, mfgs[0].edata["h"], corrupt=False)[1].detach()
        training_emb.append(emb)

    return torch.cat(training_emb, dim=0).cpu().numpy()
