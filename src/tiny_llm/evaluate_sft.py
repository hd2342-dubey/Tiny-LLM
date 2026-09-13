"""
Run a fixed set of evaluation questions through the SFT model and
print the generated answers. This is a qualitative smoke test, not a
scored benchmark — see the README's "Limited evaluation" section.

Run with:
    python -m tiny_llm.evaluate_sft
"""

from tiny_llm.checkpoint_utils import load_model, load_tokenizer
from tiny_llm.config import get_device
from tiny_llm.generation import generate_chat_reply

QUESTIONS = [
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
    "What is classification?",
]


def main():
    device = get_device()
    print("Device:", device)

    tokenizer = load_tokenizer()
    print("Vocabulary size:", tokenizer.vocab_size)

    model = load_model("sft", tokenizer, device=device)
    print("SFT checkpoint loaded.")

    print("\n" + "=" * 60)
    print("SFT EVALUATION")
    print("=" * 60)

    for index, question in enumerate(QUESTIONS, start=1):
        result = generate_chat_reply(model, tokenizer, question, device=device)

        print(f"\nExample {index}")
        print("Question:", question)
        print("Generated:", result["reply"])

    print("\n" + "=" * 60)
    print("Evaluation completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
