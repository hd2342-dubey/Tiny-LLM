"""
Phase 2 — Pretraining.

Trains TinyLLM on next-token prediction over `data/train.txt`, using a
vocabulary shared with the instruction data so the same tokenizer can
be reused during SFT. Saves the model and tokenizer to `checkpoints/`.

Run with:
    python -m tiny_llm.train
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tiny_llm.checkpoint_utils import build_model
from tiny_llm.config import (
    DEFAULT_CONFIG,
    INSTRUCTIONS_PATH,
    PRETRAINED_CHECKPOINT,
    TOKENIZER_CHECKPOINT,
    TRAIN_TEXT_PATH,
    get_device,
)
from tiny_llm.dataset import LanguageModelDataset
from tiny_llm.tokenizer import SimpleTokenizer


def main():
    config = DEFAULT_CONFIG
    device = get_device()
    print("Device:", device)

    # ------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------

    pretraining_text = TRAIN_TEXT_PATH.read_text(encoding="utf-8")
    instruction_text = INSTRUCTIONS_PATH.read_text(encoding="utf-8")

    # ------------------------------------------------------------
    # Build FINAL shared vocabulary
    # ------------------------------------------------------------

    combined_text = pretraining_text + "\n" + instruction_text
    tokenizer = SimpleTokenizer(combined_text)
    print("Vocabulary size:", tokenizer.vocab_size)

    # ------------------------------------------------------------
    # Tokenize ONLY pretraining data
    # ------------------------------------------------------------

    token_ids = tokenizer.encode(pretraining_text)
    print("Total pretraining tokens:", len(token_ids))

    # ------------------------------------------------------------
    # Dataset / DataLoader
    # ------------------------------------------------------------

    dataset = LanguageModelDataset(
        token_ids=token_ids,
        sequence_length=config.pretrain_sequence_length
    )
    print("Dataset size:", len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=config.pretrain_batch_size,
        shuffle=True
    )

    # ------------------------------------------------------------
    # Model / optimizer
    # ------------------------------------------------------------

    model = build_model(tokenizer.vocab_size, config, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.pretrain_lr
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    for epoch in range(config.pretrain_epochs):

        total_loss = 0.0

        for input_tokens, target_tokens in dataloader:

            input_tokens = input_tokens.to(device)
            target_tokens = target_tokens.to(device)

            logits = model(input_tokens)

            loss = F.cross_entropy(
                logits.view(-1, tokenizer.vocab_size),
                target_tokens.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{config.pretrain_epochs} - Loss: {average_loss:.4f}")

    # ------------------------------------------------------------
    # Save checkpoints
    # ------------------------------------------------------------

    PRETRAINED_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), PRETRAINED_CHECKPOINT)
    print(f"\nModel saved to: {PRETRAINED_CHECKPOINT}")

    tokenizer.save(TOKENIZER_CHECKPOINT)
    print(f"Tokenizer saved to: {TOKENIZER_CHECKPOINT}")


if __name__ == "__main__":
    main()
