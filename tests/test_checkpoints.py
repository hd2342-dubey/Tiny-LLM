import pytest

from tiny_llm.checkpoint_utils import load_model, load_tokenizer
from tiny_llm.config import PRETRAINED_CHECKPOINT, SFT_CHECKPOINT
from tiny_llm.generation import generate_chat_reply

pytestmark = pytest.mark.skipif(
    not (PRETRAINED_CHECKPOINT.exists() and SFT_CHECKPOINT.exists()),
    reason="trained checkpoints are not present — run `tiny-llm-train` and "
           "`tiny-llm-train-sft` first",
)


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


def test_pretrained_checkpoint_loads(tokenizer):
    model = load_model("pretrained", tokenizer)
    assert model.lm_head.out_features == tokenizer.vocab_size


def test_sft_checkpoint_loads(tokenizer):
    model = load_model("sft", tokenizer)
    assert model.lm_head.out_features == tokenizer.vocab_size


def test_sft_model_generates_a_nonempty_reply(tokenizer):
    model = load_model("sft", tokenizer)

    result = generate_chat_reply(
        model, tokenizer, "What is machine learning?", max_new_tokens=15
    )

    assert isinstance(result["reply"], str)
    assert len(result["reply"]) > 0
