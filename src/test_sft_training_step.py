import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.tokenizer.tokenizer import SimpleTokenizer
from src.instruction_dataset import InstructionDataset
from src.sft_collate import sft_collate_fn
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

tokenizer = SimpleTokenizer.__new__(SimpleTokenizer)

tokenizer.stoi = tokenizer_data["stoi"]
tokenizer.itos = tokenizer_data["itos"]

vocab_size = len(tokenizer.stoi)

print("Vocabulary size:", vocab_size)


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
# Create SFT dataset
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
    shuffle=True,
    collate_fn=sft_collate_fn
)


# ==========================================
# Load pretrained model
# ==========================================

model = TinyLLM(
    vocab_size=vocab_size,
    embedding_dim=128,
    num_heads=4,
    ff_hidden_dim=512,
    num_layers=4,
    max_sequence_length=128
).to(device)


checkpoint_path = "checkpoints/tiny_llm_pretrained.pt"

state_dict = torch.load(
    checkpoint_path,
    map_location=device
)

model.load_state_dict(state_dict)

print("Pretrained checkpoint loaded.")


# ==========================================
# Optimizer
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)


# ==========================================
# Get one batch
# ==========================================

input_tokens, target_tokens = next(
    iter(dataloader)
)

input_tokens = input_tokens.to(device)
target_tokens = target_tokens.to(device)


print("\nInput batch shape:", input_tokens.shape)
print("Target batch shape:", target_tokens.shape)


# ==========================================
# Forward pass
# ==========================================

logits = model(input_tokens)

print("Logits shape:", logits.shape)


# ==========================================
# Calculate masked loss
# ==========================================

loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    target_tokens.view(-1),
    ignore_index=-100
)

print("SFT loss before backward:", loss.item())


# ==========================================
# Backward pass
# ==========================================

optimizer.zero_grad()

loss.backward()


# ==========================================
# Check gradient
# ==========================================

gradient = model.embedding.token_embedding.weight.grad

print("Gradient exists:", gradient is not None)

print("Gradient shape:", gradient.shape)

print("Gradient mean:", gradient.mean().item())


# ==========================================
# Update weights
# ==========================================

optimizer.step()

print("SFT weights updated successfully.")