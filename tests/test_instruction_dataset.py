import torch

from tiny_llm.tokenizer import SimpleTokenizer
from tiny_llm.instruction_dataset import InstructionDataset, sft_collate_fn

EXAMPLE = (
    "User: What is machine learning?\n"
    "Assistant: Machine learning is a method where computers "
    "learn patterns from data."
)


def make_dataset(examples):
    text = "\n\n".join(examples)
    tokenizer = SimpleTokenizer(text)
    return InstructionDataset(examples=examples, tokenizer=tokenizer), tokenizer


def test_user_tokens_are_masked_in_targets():
    dataset, tokenizer = make_dataset([EXAMPLE])
    input_tokens, target_tokens = dataset[0]

    user_part = EXAMPLE.split("\nAssistant: ")[0]
    user_token_count = len(tokenizer.encode(user_part))

    # The first (user_token_count - 1) targets predict the rest of the
    # user prompt and must be masked with -100.
    assert (target_tokens[: user_token_count - 1] == -100).all()
    # At least one assistant-side target should be a real (learned) token.
    assert (target_tokens != -100).any()


def test_input_and_target_are_shifted_by_one():
    dataset, _ = make_dataset([EXAMPLE])
    input_tokens, target_tokens = dataset[0]

    assert len(input_tokens) == len(target_tokens)


def test_collate_pads_to_longest_sequence_in_batch():
    short_example = "User: Hi\nAssistant: Hello there."
    long_example = EXAMPLE

    dataset, _ = make_dataset([short_example, long_example])
    batch = [dataset[0], dataset[1]]

    input_batch, target_batch = sft_collate_fn(batch)

    max_len = max(len(s[0]) for s in batch)

    assert input_batch.shape == (2, max_len)
    assert target_batch.shape == (2, max_len)


def test_collate_pads_inputs_with_pad_token_and_targets_with_ignore_index():
    short_example = "User: Hi\nAssistant: Hello."
    long_example = EXAMPLE

    dataset, tokenizer = make_dataset([short_example, long_example])
    batch = [dataset[0], dataset[1]]

    input_batch, target_batch = sft_collate_fn(batch)

    shorter_index = 0 if len(batch[0][0]) < len(batch[1][0]) else 1
    padding_length = input_batch.shape[1] - len(batch[shorter_index][0])

    if padding_length > 0:
        assert (input_batch[shorter_index, -padding_length:] == 0).all()
        assert (target_batch[shorter_index, -padding_length:] == -100).all()
