import torch

from src.tokenizer.tokenizer import SimpleTokenizer
from src.instruction_dataset import InstructionDataset


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


# ==========================================
# Create examples
# ==========================================

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
# Inspect first example
# ==========================================

input_tokens, target_tokens = dataset[0]

print("Number of examples:", len(dataset))

print("\nInput shape:")
print(input_tokens.shape)

print("\nTarget shape:")
print(target_tokens.shape)

print("\nInput tokens:")
print(input_tokens.tolist())

print("\nTarget tokens:")
print(target_tokens.tolist())

print("\nIgnored targets:")
print(target_tokens.tolist().count(-100))

print(
    "Learning targets:",
    (target_tokens != -100).sum().item()
)