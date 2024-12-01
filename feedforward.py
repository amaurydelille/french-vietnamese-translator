import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff:int, dropout: float) -> None:
        super().__init__()
        self.first_linear_layer = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.second_linear_layer = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.second_linear_layer(self.dropout(torch.relu(self.first_linear_layer)))
