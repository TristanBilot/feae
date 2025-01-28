import torch
import torch.nn as nn
import torch.nn.functional as F

"""Some losses for few-shot head.
"""
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)

        return focal_loss

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        # self.weights = weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        class_weights = torch.tensor([1.0, 1.0])  # Initialize with equal weights

        # Assuming 'targets' is a tensor containing the target labels
        num_samples = len(targets)
        num_class_0 = (targets == 0).sum().item()
        num_class_1 = (targets == 1).sum().item()

        # Calculate class frequencies
        class_0_freq = num_class_0 / num_samples
        class_1_freq = num_class_1 / num_samples

        # Update class weights based on inverse class frequencies
        class_weights[0] = 1.0 / class_0_freq
        class_weights[1] = 1.0 / class_1_freq


        ce_loss = F.cross_entropy(inputs, targets, weight=class_weights, reduction=self.reduction)
        return ce_loss

class ClassBalancedLoss(nn.Module):
    def __init__(self, beta, num_classes, targets):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.num_classes = num_classes
        self.class_weights = self.calculate_class_weights(targets)

    def calculate_class_weights(self, targets):
        class_counts = torch.zeros(self.num_classes)
        class_weights = torch.zeros(self.num_classes)
        for class_label in range(self.num_classes):
            class_counts[class_label] = (targets == class_label).sum().item()
        effective_num = 1.0 - torch.pow(self.beta, class_counts)
        class_weights = (1.0 - self.beta) / effective_num
        return class_weights

    def forward(self, logits, targets):
        weights = self.class_weights[targets]
        loss = F.cross_entropy(logits, targets, weight=weights)
        return loss
