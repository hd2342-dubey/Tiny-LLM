"""
Phase 3 — Supervised fine-tuning (SFT).

Loads the pretrained checkpoint and fine-tunes it on instruction/
response pairs from `data/instructions.txt`, using response-only loss
masking (see `instruction_dataset.py`). Saves the result to
`checkpoints/tiny_llm_sft.pt`.

Run with:
    python -m tiny_llm.train_sft
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tiny_llm.checkpoint_utils import build_model, load_tokenizer
from tiny_llm.config import (
    DEFAULT_CONFIG,
    INSTRUCTIONS_PATH,
    PRETRAINED_CHECKPOINT,
    SFT_CHECKPOINT,
    get_device,
)
from tiny_llm.instruction_dataset import InstructionDataset, sft_collate_fn


def main():
    config = DEFAULT_CONFIG
    device = get_device()
    print("Device:", device)

    # ------------------------------------------------------------
    # Load tokenizer (shared vocabulary saved during pretraining)
    # ------------------------------------------------------------

    tokenizer = load_tokenizer()
    print("Vocabulary size:", tokenizer.vocab_size)

    # ------------------------------------------------------------
    # Load instruction data
    # ------------------------------------------------------------

    text = INSTRUCTIONS_PATH.read_text(encoding="utf-8")

    examples = [
        example.strip()
        for example in text.split("\n\n")
        if example.strip()
    ]
    print("Number of instruction examples:", len(examples))

    dataset = InstructionDataset(examples=examples, tokenizer=tokenizer)
    print("SFT dataset size:", len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=config.sft_batch_size,
        shuffle=True,
        collate_fn=sft_collate_fn
    )

    # ------------------------------------------------------------
    # Model: start from the pretrained checkpoint
    # ------------------------------------------------------------

    model = build_model(tokenizer.vocab_size, config, device)

    state_dict = torch.load(PRETRAINED_CHECKPOINT, map_location=device)
    model.load_state_dict(state_dict)
    print("Pretrained checkpoint loaded.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.sft_lr)

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    for epoch in range(config.sft_epochs):

        model.train()
        total_loss = 0.0

        for input_tokens, target_tokens in dataloader:

            input_tokens = input_tokens.to(device)
            target_tokens = target_tokens.to(device)

            logits = model(input_tokens)

            loss = F.cross_entropy(
                logits.view(-1, tokenizer.vocab_size),
                target_tokens.view(-1),
                ignore_index=-100
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{config.sft_epochs} - SFT Loss: {average_loss:.4f}")

    # ------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------

    SFT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), SFT_CHECKPOINT)
    print("\nSFT training completed.")
    print("SFT checkpoint saved to:", SFT_CHECKPOINT)


if __name__ == "__main__":
    main()
