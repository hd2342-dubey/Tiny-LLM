import torch
from torch.utils.data import DataLoader

from src.tokenizer.tokenizer import SimpleTokenizer
from src.dataset import LanguageModelDataset


# --------------------------------
# 1. Load training text
# --------------------------------

with open("data/train.txt", "r", encoding="utf-8") as f:
    text = f.read()


# --------------------------------
# 2. Create tokenizer
# --------------------------------

tokenizer = SimpleTokenizer(text)

print("Vocabulary size:", len(tokenizer.stoi))


# --------------------------------
# 3. Tokenize entire corpus
# --------------------------------

token_ids = tokenizer.encode(text)

print("Total tokens:", len(token_ids))


# --------------------------------
# 4. Create dataset
# --------------------------------

sequence_length = 32

dataset = LanguageModelDataset(
    token_ids=token_ids,
    sequence_length=sequence_length
)

print("Dataset size:", len(dataset))


# --------------------------------
# 5. Create DataLoader
# --------------------------------

batch_size = 4

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)


# --------------------------------
# 6. Inspect one batch
# --------------------------------

input_tokens, target_tokens = next(iter(dataloader))

print("\nInput batch shape:", input_tokens.shape)
print("Target batch shape:", target_tokens.shape)

print("\nFirst input:")
print(input_tokens[0])

print("\nFirst target:")
print(target_tokens[0])