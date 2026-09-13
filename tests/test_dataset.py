import torch

from tiny_llm.dataset import LanguageModelDataset


def test_dataset_length():
    token_ids = [10, 20, 30, 40, 50, 60, 70]
    dataset = LanguageModelDataset(token_ids=token_ids, sequence_length=4)

    assert len(dataset) == len(token_ids) - 4


def test_target_is_input_shifted_by_one():
    token_ids = [10, 20, 30, 40, 50, 60, 70]
    dataset = LanguageModelDataset(token_ids=token_ids, sequence_length=4)

    input_tokens, target_tokens = dataset[0]

    assert input_tokens.tolist() == [10, 20, 30, 40]
    assert target_tokens.tolist() == [20, 30, 40, 50]
    assert isinstance(input_tokens, torch.Tensor)
    assert input_tokens.dtype == torch.long
