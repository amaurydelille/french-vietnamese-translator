import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    """
    Layer normalization is a technique to normalize the activations of
    a layer across the feature dimension, it focuses on individual samples.
    It handles variable batch sizes and reduces internal covariate shifts.
    """

    def __init__(self, epsilon: float = torch.pow(10, -6)):
        # epsilon ensures we don't divide by zero because
        # it will be added to our denominator.
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.epsilon) + self.bias