"""
Inspect the pretraining data pipeline: tokenize `data/train.txt`,
build a dataset/dataloader, and print the shape of one batch.

Run with:
    python -m tiny_llm.prepare_data
"""

from torch.utils.data import DataLoader

from tiny_llm.config import DEFAULT_CONFIG, TRAIN_TEXT_PATH
from tiny_llm.dataset import LanguageModelDataset
from tiny_llm.tokenizer import SimpleTokenizer


def main():
    config = DEFAULT_CONFIG

    text = TRAIN_TEXT_PATH.read_text(encoding="utf-8")

    tokenizer = SimpleTokenizer(text)
    print("Vocabulary size:", tokenizer.vocab_size)

    token_ids = tokenizer.encode(text)
    print("Total tokens:", len(token_ids))

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

    input_tokens, target_tokens = next(iter(dataloader))

    print("\nInput batch shape:", input_tokens.shape)
    print("Target batch shape:", target_tokens.shape)

    print("\nFirst input:")
    print(input_tokens[0])

    print("\nFirst target:")
    print(target_tokens[0])


if __name__ == "__main__":
    main()
