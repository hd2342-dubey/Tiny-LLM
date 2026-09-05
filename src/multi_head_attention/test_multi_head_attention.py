import torch

from multi_head_attention import MultiHeadCausalSelfAttention


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


batch_size = 2
sequence_length = 8
embedding_dim = 128
num_heads = 4


x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim,
    device=device
)


attention = MultiHeadCausalSelfAttention(
    embedding_dim=embedding_dim,
    num_heads=num_heads
).to(device)


output = attention(x)


print("Device:", output.device)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Head dimension:", attention.head_dim)