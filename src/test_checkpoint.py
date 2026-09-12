import torch

from src.tiny_llm import TinyLLM


# ==========================================
# Device
# ==========================================

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# Load tokenizer
# ==========================================

tokenizer_data = torch.load(
    "checkpoints/tokenizer.pt",
    map_location="cpu"
)

stoi = tokenizer_data["stoi"]
itos = tokenizer_data["itos"]

vocab_size = len(stoi)

print("Tokenizer vocabulary size:", vocab_size)


# ==========================================
# Create model
# ==========================================

model = TinyLLM(
    vocab_size=vocab_size,
    embedding_dim=128,
    num_heads=4,
    ff_hidden_dim=512,
    num_layers=4,
    max_sequence_length=128
).to(device)


# ==========================================
# Load checkpoint
# ==========================================

checkpoint_path = "checkpoints/tiny_llm_pretrained.pt"

state_dict = torch.load(
    checkpoint_path,
    map_location=device
)

model.load_state_dict(state_dict)

model.eval()


# ==========================================
# Verification
# ==========================================

print("Model vocabulary size:", model.lm_head.out_features)
print("Model checkpoint loaded successfully.")