import torch

from src.self_attention.self_attention import CausalSelfAttention


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

batch_size = 2
sequence_length = 8
embedding_dim = 128


x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim,
    device=device
)


attention = CausalSelfAttention(
    embedding_dim=embedding_dim
).to(device)


output = attention(x)


print("Device:", output.device)
print("Input shape:", x.shape)
print("Output shape:", output.shape)