import torch
from torch.utils.data import Dataset

class LanguageModelDataset(Dataset):

    def __init__(self, token_ids, sequence_length):
        self.token_ids = token_ids
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.token_ids) - self.sequence_length

    def __getitem__(self, index):

        input_tokens = self.token_ids[
            index : index + self.sequence_length
        ]

        target_tokens = self.token_ids[
            index + 1 : index + self.sequence_length + 1
        ]

        return (
            torch.tensor(input_tokens, dtype=torch.long),
            torch.tensor(target_tokens, dtype=torch.long)
        )