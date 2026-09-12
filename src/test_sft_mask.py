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
# Example
# ==========================================

example = (
    "User: What is machine learning?\n"
    "Assistant: Machine learning is a method where computers "
    "learn patterns from data."
)


# ==========================================
# Separate user and assistant
# ==========================================

user_part, assistant_part = example.split(
    "\nAssistant: ",
    maxsplit=1
)

assistant_part = "Assistant: " + assistant_part


# ==========================================
# Tokenize both parts
# ==========================================

user_tokens = tokenizer.encode(user_part)

assistant_tokens = tokenizer.encode(assistant_part)


# ==========================================
# Combine tokens
# ==========================================

input_tokens = user_tokens + assistant_tokens

target_tokens = [-100] * len(user_tokens)

target_tokens += assistant_tokens


# ==========================================
# Display
# ==========================================

print("User tokens:")
print(user_tokens)

print("\nAssistant tokens:")
print(assistant_tokens)

print("\nInput tokens:")
print(input_tokens)

print("\nTarget tokens:")
print(target_tokens)

print("\nInput length:", len(input_tokens))
print("Target length:", len(target_tokens))

print(
    "\nIgnored tokens:",
    target_tokens.count(-100)
)

print(
    "Learning tokens:",
    len(target_tokens) - target_tokens.count(-100)
)