import torch
from src.embeddings.embeddings import TokenAndPositionEmbedding

# Example configuration
vocab_size = 10000
embedding_dim = 128
max_sequence_length = 128

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

embedding_layer = TokenAndPositionEmbedding(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    max_sequence_length=max_sequence_length
).to(device)


# Batch of 2 sentences
token_ids = torch.tensor(
    [
        [10, 25, 91, 42],
        [7, 18, 63, 5]
    ],
    dtype=torch.long
).to(device)


output = embedding_layer(token_ids)

print("Device:", output.device)
print("Input shape:", token_ids.shape)
print("Output shape:", output.shape)