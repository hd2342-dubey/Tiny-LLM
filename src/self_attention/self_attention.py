import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):

        # x shape:
        # [batch_size, sequence_length, embedding_dim]

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Attention scores
        scores = Q @ K.transpose(-2, -1)

        # Scale
        scores = scores / math.sqrt(self.embedding_dim)

        # Causal mask
        sequence_length = x.size(1)

        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            )
        )

        # Future positions become -infinity
        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # Convert scores into probabilities
        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # Weighted sum of values
        attention_output = attention_weights @ V

        # Final projection
        output = self.output_projection(
            attention_output
        )

        return output