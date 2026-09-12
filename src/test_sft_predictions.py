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
# Load SFT model
# ==========================================

state_dict = torch.load(
    "checkpoints/tiny_llm_sft.pt",
    map_location=device
)

model.load_state_dict(state_dict)
model.eval()


# ==========================================
# Questions to compare
# ==========================================

questions = [
    "What is machine learning?",
    "What is RAG?",
    "What is gradient descent?",
    "What is a vector database?"
]


# ==========================================
# Inspect predictions
# ==========================================

with torch.no_grad():

    for question in questions:

        prompt = (
            "User: "
            + question
            + " Assistant:"
        )

        token_ids = tokenizer.encode(prompt)

        input_tokens = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=device
        )

        logits = model(input_tokens)

        next_token_logits = logits[:, -1, :]

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1
        )

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=5,
            dim=-1
        )

        print("\n" + "=" * 60)
        print("Question:", question)
        print("=" * 60)

        for probability, token_id in zip(
            top_probabilities[0],
            top_indices[0]
        ):

            token_id = token_id.item()

            token = tokenizer.itos.get(
                token_id,
                "<UNK>"
            )

            print(
                f"{token:20s} "
                f"{probability.item():.4f}"
            )