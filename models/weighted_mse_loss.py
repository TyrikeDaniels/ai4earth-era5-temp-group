import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        # reduction='none' computes squared error per element without averaging
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target, weights):
        # Calculate squared error
        loss = self.mse(pred, target)
        # Apply weights element-wise and take the mean
        weighted_loss = loss * weights
        return torch.mean(weighted_loss)