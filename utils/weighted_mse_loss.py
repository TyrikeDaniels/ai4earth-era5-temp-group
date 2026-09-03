"""
weighted_mse_loss.py

Weighted Mean Squared Error loss module.

This module provides a weighted MSE loss implementation for use in PyTorch
models where each element in the loss can be scaled by a corresponding weight.
"""

import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    @staticmethod
    def get_bmse_weights(y_true, thresholds=(2, 5, 10, 30), weights=(1, 2, 5, 10, 30)):
        w = torch.ones_like(y_true) * weights[0]
        for thresh, weight in zip(thresholds, weights[1:]):
            w = torch.where(y_true >= thresh, torch.full_like(w, weight), w)
        return w

    def forward(self, pred, target):
        weights = self.get_bmse_weights(target)
        loss = self.mse(pred, target)
        weighted_loss = loss * weights
        return torch.mean(weighted_loss)