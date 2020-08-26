import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss for Semantic Segmentation

    Args:
        Refer to nn.CrossEntropyLoss
    """

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, pred, label):
        loss = self.ce_loss(pred, label.long())
        return loss
