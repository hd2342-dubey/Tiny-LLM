import torch

from tiny_llm.transformer_block import TransformerBlock


def test_transformer_block_preserves_shape(device):
    batch_size, sequence_length, embedding_dim = 2, 8, 128

    block = TransformerBlock(
        embedding_dim=embedding_dim, num_heads=4, ff_hidden_dim=512
    ).to(device)

    x = torch.randn(batch_size, sequence_length, embedding_dim, device=device)
    output = block(x)

    assert output.shape == x.shape


def test_residual_connections_change_the_input():
    # A block with random weights should not be a no-op / identity function
    torch.manual_seed(0)
    block = TransformerBlock(embedding_dim=16, num_heads=4, ff_hidden_dim=32)
    x = torch.randn(1, 4, 16)

    output = block(x)

    assert not torch.allclose(output, x)
