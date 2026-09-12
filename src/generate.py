import torch

from src.tokenizer.tokenizer import SimpleTokenizer
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
# Load SFT checkpoint
# ==========================================

checkpoint_path = "checkpoints/tiny_llm_sft.pt"

state_dict = torch.load(
    checkpoint_path,
    map_location=device
)

model.load_state_dict(state_dict)

model.eval()

print("SFT checkpoint loaded.")


# ==========================================
# Prompt
# ==========================================

prompt = "User: What is a Machine Learning? \n Assistant:"

token_ids = tokenizer.encode(prompt)

input_tokens = torch.tensor(
    [token_ids],
    dtype=torch.long,
    device=device
)


# ==========================================
# Generate
# ==========================================

max_new_tokens = 20

with torch.no_grad():

    for _ in range(max_new_tokens):

        logits = model(input_tokens)

        # Get logits for the final position
        next_token_logits = logits[:, -1, :]

        # Select the token with highest probability
        next_token = torch.argmax(
            next_token_logits,
            dim=-1
        )

        # Add token to sequence
        input_tokens = torch.cat(
            [
                input_tokens,
                next_token.unsqueeze(0)
            ],
            dim=1
        )


# ==========================================
# Decode
# ==========================================

generated_ids = input_tokens[0].tolist()

generated_text = tokenizer.decode(
    generated_ids
)

print("\nGenerated text:")
print(generated_text)