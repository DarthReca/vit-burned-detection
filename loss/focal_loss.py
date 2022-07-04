import torch.nn as nn
import torchvision.ops as ops


class FocalLoss(nn.Module):
    """
    Focal loss embedding implementation of torchvision. This includes sigmoid.

    Arguments
    ---------
    alpha : float, optional
        Balance for positive vs negative samples in range [0, 1].
    gamma : float, optional
        Balance for easy vs hard samples.
    reduction: str, optional
        How to make reduction (sum, mean, none)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        return ops.sigmoid_focal_loss(
            inputs, targets, self.alpha, self.gamma, self.reduction
        )
