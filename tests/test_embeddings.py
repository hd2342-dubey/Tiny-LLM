import torch

from tiny_llm.embeddings import TokenAndPositionEmbedding


def test_output_shape_matches_input(device):
    vocab_size, embedding_dim, max_sequence_length = 10_000, 128, 128

    layer = TokenAndPositionEmbedding(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_sequence_length=max_sequence_length,
    ).to(device)

    token_ids = torch.tensor(
        [[10, 25, 91, 42], [7, 18, 63, 5]], dtype=torch.long, device=device
    )

    output = layer(token_ids)

    assert output.shape == (2, 4, embedding_dim)


def test_same_token_gets_different_vectors_at_different_positions(device):
    layer = TokenAndPositionEmbedding(
        vocab_size=50, embedding_dim=16, max_sequence_length=8
    ).to(device)

    # Token id 3 repeated at two different positions
    token_ids = torch.tensor([[3, 3]], dtype=torch.long, device=device)
    output = layer(token_ids)

    # Position embeddings differ, so the two output vectors must differ
    assert not torch.allclose(output[0, 0], output[0, 1])
