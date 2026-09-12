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

print("Number of instruction examples:", len(examples))


# ==========================================
# Create SFT dataset
# ==========================================

dataset = InstructionDataset(
    examples=examples,
    tokenizer=tokenizer
)

print("SFT dataset size:", len(dataset))


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
# Load pretrained model
# ==========================================

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
# Training
# ==========================================

epochs = 100

for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for input_tokens, target_tokens in dataloader:

        input_tokens = input_tokens.to(device)
        target_tokens = target_tokens.to(device)

        # ------------------------------
        # Forward pass
        # ------------------------------

        logits = model(input_tokens)

        # ------------------------------
        # Calculate masked loss
        # ------------------------------

        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            target_tokens.view(-1),
            ignore_index=-100
        )

        # ------------------------------
        # Backpropagation
        # ------------------------------

        optimizer.zero_grad()

        loss.backward()

        # ------------------------------
        # Update parameters
        # ------------------------------

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- SFT Loss: {average_loss:.4f}"
    )


# ==========================================
# Save SFT checkpoint
# ==========================================

output_path = "checkpoints/tiny_llm_sft.pt"

torch.save(
    model.state_dict(),
    output_path
)

print("\nSFT training completed.")
print("SFT checkpoint saved to:", output_path)