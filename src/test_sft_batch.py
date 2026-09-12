import torch
from torch.utils.data import DataLoader

from src.tokenizer.tokenizer import SimpleTokenizer
from src.instruction_dataset import InstructionDataset
from src.sft_collate import sft_collate_fn


# ==========================================
# Load tokenizer
# ==========================================

tokenizer_data = torch.load(
    "checkpoints/tokenizer.pt",
    map_location="cpu"
)

tokenizer = SimpleTokenizer.__new__(SimpleTokenizer)

tokenizer.stoi = tokenizer_data["stoi"]
tokenizer.itos = tokenizer_data["itos"]


# ==========================================
# Load instruction data
# ==========================================

with open(
    "data/instructions.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()


examples = [
    example.strip()
    for example in text.split("\n\n")
    if example.strip()
]


# ==========================================
# Create dataset
# ==========================================

dataset = InstructionDataset(
    examples=examples,
    tokenizer=tokenizer
)


# ==========================================
# Create DataLoader
# ==========================================

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=sft_collate_fn
)


# ==========================================
# Get first batch
# ==========================================

input_tokens, target_tokens = next(
    iter(dataloader)
)


# ==========================================
# Inspect batch
# ==========================================

print("Input batch shape:")
print(input_tokens.shape)

print("\nTarget batch shape:")
print(target_tokens.shape)

print("\nInput batch:")
print(input_tokens)

print("\nTarget batch:")
print(target_tokens)

print(
    "\nPadding tokens in input:",
    (input_tokens == 0).sum().item()
)

print(
    "Ignored targets:",
    (target_tokens == -100).sum().item()
)