import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, sequence_length: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length

        # nn.Dropout acts like the Dropout layer from keras,
        # it reduces overfitting by randomly disabling neurons: 
        # some inputs units are set to zero following a probability p.
        # by preventilg reliance on specific neurons, dropout encourages
        # the network to learn robust and distributed representations.
        self.dropout = nn.Dropout(dropout)

        positional_encoding = torch.zeros(sequence_length, d_model)
        # unsqueeze adds a dimension and reshapes the tensor.
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1 )
        div_term = torch.exp(torch.arrange(0, d_model, 2).float() * (10000 / d_model))

        # applying positional encoding formula to each terms.
        positional_encoding[:, 0::2] = torch.sin(position, div_term)
        position[:, 1::2] = torch.cos(position, div_term)

        positional_encoding = positional_encoding.unsqueeze(0)

        # saves the tensor.
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        # requires_grad_ ensures that gradients are not computed 
        # during backprop, because positional encoding is usually
        # static and not learned.
        x += (self.positional_encoding[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
    
        