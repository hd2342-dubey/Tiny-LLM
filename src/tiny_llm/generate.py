"""
Generate a single response from the fine-tuned (SFT) model on the
command line.

Run with:
    python -m tiny_llm.generate "What is machine learning?"
"""

import argparse

from tiny_llm.checkpoint_utils import load_model, load_tokenizer
from tiny_llm.config import get_device
from tiny_llm.generation import generate_chat_reply


def main():
    parser = argparse.ArgumentParser(description="Generate a response from TinyLLM.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="What is a Machine Learning?",
        help="The user message to send to the model.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument(
        "--checkpoint",
        choices=["pretrained", "sft"],
        default="sft",
    )
    args = parser.parse_args()

    device = get_device()
    print("Device:", device)

    tokenizer = load_tokenizer()
    print("Vocabulary size:", tokenizer.vocab_size)

    model = load_model(args.checkpoint, tokenizer, device=device)
    print(f"{args.checkpoint} checkpoint loaded.")

    result = generate_chat_reply(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )

    print("\nPrompt:", args.prompt)
    print("Generated:", result["reply"])


if __name__ == "__main__":
    main()
