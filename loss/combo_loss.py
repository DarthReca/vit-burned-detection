# Copyrigth 2022 Daniele Rege Cambrin
from typing import Any, Dict, List

import torch.nn as nn
import utils


class ComboLoss(nn.Module):
    """
    ComboLoss multiply each loss by its weight and them sum up

    Attributes
    ----------
    losses : Dict[str, Dict[str, Any]]
        Dict with loss name and relative parameters.
    weights : List[float]
        Corresponding weight for each loss.

    """

    def __init__(self, losses: Dict[str, Dict[str, Any]], weights: List[float]):
        super(ComboLoss, self).__init__()
        self.losses = nn.ModuleList(
            [utils.config_to_object("torch.nn", k, v) for k, v in losses.items()]
        )
        self.weights = weights

    def forward(self, inputs, targets):
        loss = 0
        for i in range(len(self.losses)):
            loss += self.weights[i] * self.losses[i](inputs, targets)
        return loss
