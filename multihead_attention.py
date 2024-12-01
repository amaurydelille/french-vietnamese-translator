import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_query = nn.Linear(d_model, d_model)
        self.w_key = nn.Linear(d_model, d_model)
        self.w_value = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        numerator = query @ key.transpose(-2, -1)
        denominator = torch.sqrt(d_k)
        attention_score = numerator / denominator

        if mask:
            attention_score.masked_fill_(mask == 0, -1e9)

        attention_score = attention_score.softmax(dim=-1)
        
        if dropout:
            attention_score = dropout(attention_score)

        return attention_score @ value

    def forward(self, q, k, v, mask):
        query = self.w_query(q)
        key = self.w_key(k)
        value = self.w_value(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        x, self.attention_score = MultiHeadAttention.attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)