import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.tokenizer.tokenizer import SimpleTokenizer
from src.dataset import LanguageModelDataset
from src.tiny_llm import TinyLLM


# --------------------------------
# Device
# --------------------------------

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------
# Load training text
# --------------------------------

with open("data/train.txt", "r", encoding="utf-8") as f:
    text = f.read()


# --------------------------------
# Tokenizer
# --------------------------------

tokenizer = SimpleTokenizer(text)

vocab_size = len(tokenizer.stoi)

print("Vocabulary size:", vocab_size)


# --------------------------------
# Tokenize corpus
# --------------------------------

token_ids = tokenizer.encode(text)


# --------------------------------
# Dataset
# --------------------------------

sequence_length = 32

dataset = LanguageModelDataset(
    token_ids=token_ids,
    sequence_length=sequence_length
)


# --------------------------------
# DataLoader
# --------------------------------

batch_size = 4

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)


# --------------------------------
# Model
# --------------------------------

model = TinyLLM(
    vocab_size=vocab_size,
    embedding_dim=128,
    num_heads=4,
    ff_hidden_dim=512,
    num_layers=4,
    max_sequence_length=128
).to(device)


# --------------------------------
# Optimizer
# --------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)


# --------------------------------
# Get one batch
# --------------------------------

input_tokens, target_tokens = next(iter(dataloader))

input_tokens = input_tokens.to(device)
target_tokens = target_tokens.to(device)


# --------------------------------
# Forward pass
# --------------------------------

logits = model(input_tokens)


# --------------------------------
# Calculate loss
# --------------------------------

loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    target_tokens.view(-1)
)

print("Loss before backward:", loss.item())


# --------------------------------
# Backpropagation
# --------------------------------

optimizer.zero_grad()

loss.backward()


# --------------------------------
# Inspect a gradient
# --------------------------------

gradient = model.embedding.token_embedding.weight.grad

print("Gradient exists:", gradient is not None)
print("Gradient shape:", gradient.shape)
print("Gradient mean:", gradient.mean().item())


# --------------------------------
# Update model weights
# --------------------------------

optimizer.step()

print("Weights updated successfully.")