import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding_layer = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding_layer(x) * torch.sqrt(self.d_model)