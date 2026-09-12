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

state_dict = torch.load(
    "checkpoints/tiny_llm_sft.pt",
    map_location=device
)

model.load_state_dict(state_dict)

model.eval()

print("SFT checkpoint loaded.")


# ==========================================
# Evaluation questions
# ==========================================

questions = [
    "What is artificial intelligence?",
    "What is machine learning?",
    "What is deep learning?",
    "What is a neural network?",
    "What is a transformer?",
    "What is self attention?",
    "What is a vector database?",
    "What is RAG?",
    "What is an embedding?",
    "What is gradient descent?",
    "What is overfitting?",
    "What is classification?"
]


# ==========================================
# Generate answer
# ==========================================

def generate_answer(question, max_new_tokens=20):

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

    with torch.no_grad():

        for _ in range(max_new_tokens):

            logits = model(input_tokens)

            next_token_logits = logits[:, -1, :]

            next_token = torch.argmax(
                next_token_logits,
                dim=-1
            )

            input_tokens = torch.cat(
                [
                    input_tokens,
                    next_token.unsqueeze(0)
                ],
                dim=1
            )

    generated_ids = input_tokens[0].tolist()

    generated_text = tokenizer.decode(
        generated_ids
    )

    # Return only the part after Assistant:
    answer = generated_text.split(
        "Assistant:",
        maxsplit=1
    )[1]

    return answer.strip()


# ==========================================
# Run evaluation
# ==========================================

print("\n" + "=" * 60)
print("SFT EVALUATION")
print("=" * 60)

for index, question in enumerate(questions, start=1):

    answer = generate_answer(question)

    print(f"\nExample {index}")
    print("Question:", question)
    print("Generated:", answer)

print("\n" + "=" * 60)
print("Evaluation completed.")
print("=" * 60)