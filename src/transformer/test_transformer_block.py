import torch
from src.transformer.transformer_block import TransformerBlock

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


batch_size = 2
sequence_length = 8
embedding_dim = 128
num_heads = 4
ff_hidden_dim = 512


x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim,
    device=device
)


block = TransformerBlock(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    ff_hidden_dim=ff_hidden_dim
).to(device)


output = block(x)


print("Device:", output.device)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
print(
    "Shape preserved:",
    x.shape == output.shape
)