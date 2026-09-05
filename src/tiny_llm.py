import torch
import torch.nn as nn

from src.embeddings.embeddings import TokenAndPositionEmbedding
from src.transformer.transformer_block import TransformerBlock


class TinyLLM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_heads=4,
        ff_hidden_dim=512,
        num_layers=4,
        max_sequence_length=128
    ):
        super().__init__()

        self.embedding = TokenAndPositionEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length
        )

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    ff_hidden_dim=ff_hidden_dim
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(
            embedding_dim
        )

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, token_ids):

        x = self.embedding(token_ids)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.final_layer_norm(x)

        logits = self.lm_head(x)

        return logits