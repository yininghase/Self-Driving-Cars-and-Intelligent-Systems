import torch.nn as nn
import torch

class WeightedMeanSquaredError(nn.Module):
    def __init__(self, batch_size, horizon, num_vehicles):
        super().__init__()
        self.batch_size = batch_size
        self.horizon = horizon
        self.num_vehicles = num_vehicles
    
    def forward(self, inputs, targets, weights):
        loss = ((inputs - targets)**2).reshape(-1, self.horizon, self.num_vehicles, 2)
        mean_loss = torch.sum(loss, axis=[2,3])
        weighted_mean_loss = torch.mul(mean_loss, weights)
        return torch.mean(weighted_mean_loss)