"""
Tiny LLM — a small, from-scratch, GPT-style causal language model
implemented in PyTorch for educational purposes.

This package contains every building block of the model (tokenizer,
embeddings, attention, transformer blocks), the datasets and training
loops used for pretraining and instruction fine-tuning (SFT), and the
generation utilities used by both the CLI scripts and the web API.
"""

from tiny_llm.config import TinyLLMConfig
from tiny_llm.model import TinyLLM
from tiny_llm.tokenizer import SimpleTokenizer

__all__ = [
    "TinyLLM",
    "TinyLLMConfig",
    "SimpleTokenizer",
]

__version__ = "2.0.0"
