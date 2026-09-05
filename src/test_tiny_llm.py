import torch

from src.tiny_llm import TinyLLM


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# Model configuration

vocab_size = 10000
embedding_dim = 128
num_heads = 4
ff_hidden_dim = 512
num_layers = 4
max_sequence_length = 128


model = TinyLLM(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    ff_hidden_dim=ff_hidden_dim,
    num_layers=num_layers,
    max_sequence_length=max_sequence_length
).to(device)


# Fake tokenized input

batch_size = 2
sequence_length = 8

token_ids = torch.randint(
    0,
    vocab_size,
    (
        batch_size,
        sequence_length
    ),
    device=device
)


# Forward pass

logits = model(token_ids)


print("Device:", logits.device)

print(
    "Input shape:",
    token_ids.shape
)

print(
    "Logits shape:",
    logits.shape
)

print(
    "Number of parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)