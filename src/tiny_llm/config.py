"""
Central configuration for Tiny LLM.

Every training / generation script previously hardcoded the same
architecture numbers (embedding_dim=128, num_heads=4, ...) in five
different places. They now all import a single `TinyLLMConfig` from
here, so changing the architecture only requires editing one file.
"""

from dataclasses import dataclass
from pathlib import Path

import torch

# ---------------------------------------------------------------------
# Project paths (resolved relative to this file, so scripts work no
# matter what directory they are run from).
# ---------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent          # src/tiny_llm
PROJECT_ROOT = PACKAGE_DIR.parent.parent                # repo root

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

TRAIN_TEXT_PATH = DATA_DIR / "train.txt"
INSTRUCTIONS_PATH = DATA_DIR / "instructions.txt"

TOKENIZER_CHECKPOINT = CHECKPOINT_DIR / "tokenizer.pt"
PRETRAINED_CHECKPOINT = CHECKPOINT_DIR / "tiny_llm_pretrained.pt"
SFT_CHECKPOINT = CHECKPOINT_DIR / "tiny_llm_sft.pt"


# ---------------------------------------------------------------------
# Model / training configuration
# ---------------------------------------------------------------------

@dataclass
class TinyLLMConfig:
    """Architecture and training hyperparameters for TinyLLM."""

    embedding_dim: int = 128
    num_heads: int = 4
    ff_hidden_dim: int = 512
    num_layers: int = 4
    max_sequence_length: int = 128

    # Pretraining
    pretrain_sequence_length: int = 32
    pretrain_batch_size: int = 4
    pretrain_lr: float = 3e-4
    pretrain_epochs: int = 20

    # Supervised fine-tuning
    sft_batch_size: int = 4
    sft_lr: float = 1e-4
    sft_epochs: int = 100


DEFAULT_CONFIG = TinyLLMConfig()


def get_device() -> torch.device:
    """Pick the best available device: CUDA > Apple MPS > CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
