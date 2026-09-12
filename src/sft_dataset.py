import torch
from torch.utils.data import Dataset


class InstructionDataset(Dataset):

    def __init__(self, examples, tokenizer, sequence_length=64):

        self.examples = examples
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

        self.samples = []

        for example in examples:

            token_ids = tokenizer.encode(example)

            # Keep only examples that fit
            # inside the sequence length.
            if len(token_ids) <= sequence_length:
                self.samples.append(token_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        token_ids = self.samples[index]

        input_tokens = token_ids[:-1]
        target_tokens = token_ids[1:]

        return (
            torch.tensor(input_tokens, dtype=torch.long),
            torch.tensor(target_tokens, dtype=torch.long)
        )