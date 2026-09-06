import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.tokenizer.tokenizer import SimpleTokenizer
from src.dataset import LanguageModelDataset
from src.tiny_llm import TinyLLM


# ==========================================
# 1. Device
# ==========================================

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# 2. Load training text
# ==========================================

with open("data/train.txt", "r", encoding="utf-8") as f:
    text = f.read()


# ==========================================
# 3. Tokenizer
# ==========================================

tokenizer = SimpleTokenizer(text)

vocab_size = len(tokenizer.stoi)

print("Vocabulary size:", vocab_size)


# ==========================================
# 4. Tokenize
# ==========================================

token_ids = tokenizer.encode(text)

print("Total tokens:", len(token_ids))


# ==========================================
# 5. Dataset
# ==========================================

sequence_length = 32

dataset = LanguageModelDataset(
    token_ids=token_ids,
    sequence_length=sequence_length
)

print("Dataset size:", len(dataset))


# ==========================================
# 6. DataLoader
# ==========================================

batch_size = 4

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)


# ==========================================
# 7. Model
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
# 8. Optimizer
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)


# ==========================================
# 9. Training
# ==========================================

epochs = 20

for epoch in range(epochs):

    total_loss = 0.0

    for input_tokens, target_tokens in dataloader:

        input_tokens = input_tokens.to(device)
        target_tokens = target_tokens.to(device)

        # -----------------------------
        # Forward pass
        # -----------------------------

        logits = model(input_tokens)

        # -----------------------------
        # Loss
        # -----------------------------

        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            target_tokens.view(-1)
        )

        # -----------------------------
        # Backpropagation
        # -----------------------------

        optimizer.zero_grad()

        loss.backward()

        # -----------------------------
        # Update weights
        # -----------------------------

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- Loss: {average_loss:.4f}"
    )

# ==========================================
# 10. Save model checkpoint
# ==========================================

model_path = "checkpoints/tiny_llm_pretrained.pt"

torch.save(
    model.state_dict(),
    model_path
)

print(f"\nModel saved to: {model_path}")


# ==========================================
# 11. Save tokenizer
# ==========================================

tokenizer_path = "checkpoints/tokenizer.pt"

torch.save(
    {
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos
    },
    tokenizer_path
)

print(f"Tokenizer saved to: {tokenizer_path}")