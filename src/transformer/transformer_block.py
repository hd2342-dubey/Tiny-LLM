import torch.nn as nn
from src.multi_head_attention.multi_head_attention import MultiHeadCausalSelfAttention


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        ff_hidden_dim
    ):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(
            embedding_dim
        )

        self.attention = MultiHeadCausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads
        )

        self.layer_norm_2 = nn.LayerNorm(
            embedding_dim
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                embedding_dim,
                ff_hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                ff_hidden_dim,
                embedding_dim
            )
        )

    def forward(self, x):

        # Pre-LayerNorm + Attention + Residual
        x = x + self.attention(
            self.layer_norm_1(x)
        )

        # Pre-LayerNorm + FFN + Residual
        x = x + self.feed_forward(
            self.layer_norm_2(x)
        )

        return x