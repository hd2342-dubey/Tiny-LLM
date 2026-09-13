import torch
import torch.nn.functional as F

from tiny_llm.model import TinyLLM


def make_model(vocab_size=1000, **overrides):
    config = dict(
        vocab_size=vocab_size,
        embedding_dim=32,
        num_heads=4,
        ff_hidden_dim=64,
        num_layers=2,
        max_sequence_length=64,
    )
    config.update(overrides)
    return TinyLLM(**config)


def test_forward_pass_shape(device):
    vocab_size = 1000
    model = make_model(vocab_size).to(device)

    batch_size, sequence_length = 2, 8
    token_ids = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)

    logits = model(token_ids)

    assert logits.shape == (batch_size, sequence_length, vocab_size)


def test_num_parameters_is_positive():
    model = make_model()
    assert model.num_parameters() > 0


def test_gradients_flow_to_embeddings(device):
    vocab_size = 200
    model = make_model(vocab_size).to(device)

    token_ids = torch.randint(0, vocab_size, (2, 6), device=device)
    targets = torch.randint(0, vocab_size, (2, 6), device=device)

    logits = model(token_ids)
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    loss.backward()

    gradient = model.embedding.token_embedding.weight.grad

    assert gradient is not None
    assert gradient.shape == model.embedding.token_embedding.weight.shape
    assert torch.isfinite(gradient).all()


def test_ignore_index_masks_loss(device):
    """Mirrors the SFT loss: padded/-100 targets shouldn't contribute."""

    vocab_size = 50
    model = make_model(vocab_size).to(device)

    token_ids = torch.randint(0, vocab_size, (1, 4), device=device)
    all_masked_targets = torch.full((1, 4), -100, dtype=torch.long, device=device)

    logits = model(token_ids)
    loss = F.cross_entropy(
        logits.view(-1, vocab_size), all_masked_targets.view(-1), ignore_index=-100
    )

    # cross_entropy over an all-ignored batch is defined as 0
    assert loss.item() == 0.0
