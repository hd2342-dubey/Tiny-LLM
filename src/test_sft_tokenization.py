import torch

from src.tokenizer.tokenizer import SimpleTokenizer


# ==========================================
# Load the same tokenizer used by the model
# ==========================================

tokenizer_data = torch.load(
    "checkpoints/tokenizer.pt",
    map_location="cpu"
)

tokenizer = SimpleTokenizer.__new__(SimpleTokenizer)

tokenizer.stoi = tokenizer_data["stoi"]
tokenizer.itos = tokenizer_data["itos"]


# ==========================================
# Test instruction
# ==========================================

text = (
    "User: What is machine learning?\n"
    "Assistant: Machine learning is a method where computers "
    "learn patterns from data."
)


# ==========================================
# Encode
# ==========================================

token_ids = tokenizer.encode(text)

print("Original text:")
print(text)

print("\nToken IDs:")
print(token_ids)

print("\nNumber of tokens:")
print(len(token_ids))


# ==========================================
# Decode
# ==========================================

decoded = tokenizer.decode(token_ids)

print("\nDecoded text:")
print(decoded)