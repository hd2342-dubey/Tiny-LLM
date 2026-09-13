import torch
import pytest

from tiny_llm.tokenizer import SimpleTokenizer


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def toy_text():
    return (
        "I am a student\n"
        "I am learning artificial intelligence\n"
        "I love deep learning\n"
    )


@pytest.fixture
def toy_tokenizer(toy_text):
    return SimpleTokenizer(toy_text)
