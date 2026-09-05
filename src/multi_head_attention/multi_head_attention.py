import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCausalSelfAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

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

        batch_size, sequence_length, _ = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # [B, T, D]
        #      ↓
        # [B, T, H, head_dim]
        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        # Move heads before sequence dimension
        #
        # [B, T, H, head_dim]
        #       ↓
        # [B, H, T, head_dim]

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention scores
        #
        # [B, H, T, head_dim]
        # ×
        # [B, H, head_dim, T]
        #
        # =
        #
        # [B, H, T, T]

        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dim)

        # Causal mask
        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        attention_output = attention_weights @ V

        # [B, H, T, head_dim]
        #       ↓
        # [B, T, H, head_dim]

        attention_output = attention_output.transpose(1, 2)

        # Make tensor contiguous before reshape
        attention_output = attention_output.contiguous()

        # [B, T, H, head_dim]
        #       ↓
        # [B, T, D]

        attention_output = attention_output.view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        output = self.output_projection(
            attention_output
        )

        return output