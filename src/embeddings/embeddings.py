import torch
import torch.nn as nn


class TokenAndPositionEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim, max_sequence_length):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            max_sequence_length,
            embedding_dim
        )

    def forward(self, token_ids):

        batch_size, sequence_length = token_ids.shape

        # Token embeddings
        token_vectors = self.token_embedding(token_ids)

        # Position IDs: 0, 1, 2, ..., sequence_length-1
        position_ids = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        # Position embeddings
        position_vectors = self.position_embedding(position_ids)

        # Add token + position information
        embeddings = token_vectors + position_vectors

        return embeddings