"""
A minimal word-level tokenizer.

This is intentionally the simplest possible tokenizer: it splits on
whitespace and assigns every unique word an integer id. It has no
subword handling, so any word not seen during vocabulary construction
becomes ``<UNK>``. See the README for why this is a known, deliberate
limitation of this educational project.
"""

from __future__ import annotations

import torch


class SimpleTokenizer:

    def __init__(self, text: str):
        words = sorted(set(text.split()))

        self.stoi = {
            "<PAD>": 0,
            "<UNK>": 1
        }

        for word in words:
            if word not in self.stoi:
                self.stoi[word] = len(self.stoi)

        self.itos = {
            i: word
            for word, i in self.stoi.items()
        }

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [
            self.stoi.get(word, self.stoi["<UNK>"])
            for word in text.split()
        ]

    def decode(self, token_ids) -> str:
        return " ".join(
            self.itos.get(token_id, "<UNK>")
            for token_id in token_ids
        )

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def state_dict(self) -> dict:
        return {"stoi": self.stoi, "itos": self.itos}

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def from_state_dict(cls, state: dict) -> "SimpleTokenizer":
        tokenizer = cls.__new__(cls)
        tokenizer.stoi = state["stoi"]
        tokenizer.itos = state["itos"]
        return tokenizer

    @classmethod
    def load(cls, path) -> "SimpleTokenizer":
        state = torch.load(path, map_location="cpu")
        return cls.from_state_dict(state)
