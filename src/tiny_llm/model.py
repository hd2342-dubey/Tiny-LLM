import torch.nn as nn

from tiny_llm.config import TinyLLMConfig
from tiny_llm.embeddings import TokenAndPositionEmbedding
from tiny_llm.transformer_block import TransformerBlock


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

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length

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

    def forward(self, token_ids, return_internals: bool = False):
        """Standard forward pass, returning just the logits.

        When ``return_internals=True``, also returns everything the
        visualizer needs to show *how* those logits were produced:
        the embedding output and, per transformer block, the
        post-block hidden state and the per-head attention weights.
        This does not change the math — it just captures tensors that
        were already being computed and discarded.
        """

        x = self.embedding(token_ids)

        if not return_internals:
            for block in self.transformer_blocks:
                x = block(x)

            x = self.final_layer_norm(x)

            return self.lm_head(x)

        embedding_output = x
        layer_hidden_states = []
        layer_attention_weights = []

        for block in self.transformer_blocks:
            x, attention_weights = block(x, return_weights=True)
            layer_hidden_states.append(x)
            layer_attention_weights.append(attention_weights)

        x = self.final_layer_norm(x)

        logits = self.lm_head(x)

        return {
            "logits": logits,
            "embedding_output": embedding_output,
            "layer_hidden_states": layer_hidden_states,
            "layer_attention_weights": layer_attention_weights,
        }

    # -----------------------------------------------------------------
    # Convenience helpers (used by the CLI scripts and the web API)
    # -----------------------------------------------------------------

    @classmethod
    def from_config(cls, vocab_size: int, config: TinyLLMConfig) -> "TinyLLM":
        return cls(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            ff_hidden_dim=config.ff_hidden_dim,
            num_layers=config.num_layers,
            max_sequence_length=config.max_sequence_length
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
