import torch
from torch.utils.data import Dataset


class InstructionDataset(Dataset):

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
                    torch.tensor(
                        input_tokens,
                        dtype=torch.long
                    ),
                    torch.tensor(
                        target_tokens,
                        dtype=torch.long
                    )
                )
            )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        return self.samples[index]