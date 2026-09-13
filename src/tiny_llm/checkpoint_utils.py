"""
Shared checkpoint loading helpers.

Previously, `generate.py`, `evaluate_sft.py`, `train_sft.py`, and (now)
the web API each duplicated the same ~15 lines to load the tokenizer
and reconstruct the model from a checkpoint. That logic now lives here
once.
"""

from __future__ import annotations

import torch

from tiny_llm.config import (
    DEFAULT_CONFIG,
    PRETRAINED_CHECKPOINT,
    SFT_CHECKPOINT,
    TOKENIZER_CHECKPOINT,
    TinyLLMConfig,
    get_device,
)
from tiny_llm.model import TinyLLM
from tiny_llm.tokenizer import SimpleTokenizer

CHECKPOINTS = {
    "pretrained": PRETRAINED_CHECKPOINT,
    "sft": SFT_CHECKPOINT,
}


def load_tokenizer(path=TOKENIZER_CHECKPOINT) -> SimpleTokenizer:
    return SimpleTokenizer.load(path)


def build_model(
    vocab_size: int,
    config: TinyLLMConfig = DEFAULT_CONFIG,
    device: torch.device | None = None,
) -> TinyLLM:
    device = device or get_device()
    return TinyLLM.from_config(vocab_size, config).to(device)


def load_model(
    checkpoint: str,
    tokenizer: SimpleTokenizer,
    config: TinyLLMConfig = DEFAULT_CONFIG,
    device: torch.device | None = None,
) -> TinyLLM:
    """Build a TinyLLM and load weights from a named checkpoint.

    ``checkpoint`` is either "pretrained" or "sft", or a direct path to
    a ``.pt`` state-dict file.
    """

    device = device or get_device()
    checkpoint_path = CHECKPOINTS.get(checkpoint, checkpoint)

    model = build_model(tokenizer.vocab_size, config, device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model
