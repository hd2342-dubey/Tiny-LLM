import torch
from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    """
    Instruction/response pairs for supervised fine-tuning.

    Each example is tokenized as ``user_tokens + assistant_tokens``.
    Target positions that fall inside the user prompt are masked with
    ``-100`` so `nn.CrossEntropyLoss` ignores them — the model is only
    trained to predict the assistant's response, not the question.
    """

    def __init__(self, examples, tokenizer):

        self.samples = []

        for example in examples:

            # ----------------------------------
            # Separate user and assistant
            # ----------------------------------

            user_part, assistant_part = example.split(
                "\nAssistant: ",
                maxsplit=1
            )

            assistant_part = "Assistant: " + assistant_part

            # ----------------------------------
            # Tokenize
            # ----------------------------------

            user_tokens = tokenizer.encode(user_part)
            assistant_tokens = tokenizer.encode(assistant_part)

            # ----------------------------------
            # Combine
            # ----------------------------------

            token_ids = user_tokens + assistant_tokens

            # ----------------------------------
            # Create next-token targets
            # ----------------------------------

            input_tokens = token_ids[:-1]
            target_tokens = token_ids[1:]

            # ----------------------------------
            # Create loss mask
            # ----------------------------------

            # The first len(user_tokens) - 1
            # target positions correspond to
            # predicting the rest of the user prompt.
            user_target_count = len(user_tokens) - 1

            target_tokens = (
                [-100] * user_target_count
                + target_tokens[user_target_count:]
            )

            # ----------------------------------
            # Store sample
            # ----------------------------------

            self.samples.append(
                (
                    torch.tensor(input_tokens, dtype=torch.long),
                    torch.tensor(target_tokens, dtype=torch.long)
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def sft_collate_fn(batch):
    """Right-pad a batch of variable-length (input, target) pairs.

    Inputs are padded with ``<PAD>`` (token id 0); targets are padded
    with ``-100`` so padding never contributes to the loss.
    """

    input_sequences = [item[0] for item in batch]
    target_sequences = [item[1] for item in batch]

    max_length = max(
        len(sequence)
        for sequence in input_sequences
    )

    padded_inputs = []
    padded_targets = []

    for input_tokens, target_tokens in zip(
        input_sequences,
        target_sequences
    ):

        padding_length = max_length - len(input_tokens)

        padded_input = torch.cat(
            [
                input_tokens,
                torch.zeros(padding_length, dtype=torch.long)
            ]
        )

        padded_target = torch.cat(
            [
                target_tokens,
                torch.full((padding_length,), -100, dtype=torch.long)
            ]
        )

        padded_inputs.append(padded_input)
        padded_targets.append(padded_target)

    return (
        torch.stack(padded_inputs),
        torch.stack(padded_targets)
    )
