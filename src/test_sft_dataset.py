import torch

from src.tokenizer.tokenizer import SimpleTokenizer


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
# Split examples
# ==========================================

examples = [
    example.strip()
    for example in text.split("\n\n")
    if example.strip()
]


# ==========================================
# Display dataset information
# ==========================================

print("Number of instruction examples:", len(examples))

print("\n")


for i, example in enumerate(examples):

    token_ids = tokenizer.encode(example)

    print(f"Example {i + 1}")
    print("--------------------")

    print(example)

    print("\nToken count:", len(token_ids))

    print("Token IDs:", token_ids)

    print()