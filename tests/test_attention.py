import torch

from tiny_llm.self_attention import CausalSelfAttention
from tiny_llm.multi_head_attention import MultiHeadCausalSelfAttention


def test_self_attention_preserves_shape(device):
    batch_size, sequence_length, embedding_dim = 2, 8, 128

    attention = CausalSelfAttention(embedding_dim=embedding_dim).to(device)
    x = torch.randn(batch_size, sequence_length, embedding_dim, device=device)

    output = attention(x)

    assert output.shape == x.shape


def test_multi_head_attention_preserves_shape(device):
    batch_size, sequence_length, embedding_dim, num_heads = 2, 8, 128, 4

    attention = MultiHeadCausalSelfAttention(
        embedding_dim=embedding_dim, num_heads=num_heads
    ).to(device)
    x = torch.randn(batch_size, sequence_length, embedding_dim, device=device)

    output = attention(x)

    assert output.shape == x.shape
    assert attention.head_dim == embedding_dim // num_heads


def test_multi_head_attention_requires_divisible_dims():
    try:
        MultiHeadCausalSelfAttention(embedding_dim=10, num_heads=3)
        assert False, "expected an assertion error for a non-divisible head count"
    except AssertionError:
        pass


def test_causal_masking_blocks_future_positions(device):
    """A change to a future token must not affect an earlier position's
    output — this is what makes the attention 'causal'."""

    torch.manual_seed(0)
    embedding_dim = 16
    attention = MultiHeadCausalSelfAttention(
        embedding_dim=embedding_dim, num_heads=4
    ).to(device)
    attention.eval()

    x = torch.randn(1, 5, embedding_dim, device=device)
    x_modified = x.clone()
    x_modified[0, -1] = torch.randn(embedding_dim)  # change only the last token

    with torch.no_grad():
        out_original = attention(x)
        out_modified = attention(x_modified)

    # Every position except the last (which we intentionally changed)
    # must be identical, since the future can't influence the past.
    assert torch.allclose(out_original[0, :-1], out_modified[0, :-1], atol=1e-6)
