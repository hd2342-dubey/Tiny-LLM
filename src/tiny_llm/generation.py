"""
Autoregressive text generation.

The original scripts each implemented the same greedy decoding loop
inline. This module extracts it once and adds optional temperature /
top-k sampling on top of the same loop, so the web UI can expose a
"creativity" control without touching the core model code.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tiny_llm.model import TinyLLM
from tiny_llm.tokenizer import SimpleTokenizer


@torch.no_grad()
def generate(
    model: TinyLLM,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 0.0,
    top_k: int | None = None,
    device: torch.device | None = None,
) -> dict:
    """Generate a continuation for ``prompt``.

    temperature=0.0 reproduces the original greedy (argmax) decoding
    used throughout the project. Any temperature > 0 samples from the
    (optionally top-k filtered) softmax distribution instead.

    Returns a dict with the full text, just the newly generated text,
    and the prompt/completion token counts (useful for the UI).
    """

    device = device or next(model.parameters()).device

    prompt_ids = tokenizer.encode(prompt)
    input_tokens = torch.tensor(
        [prompt_ids], dtype=torch.long, device=device
    )

    prompt_length = input_tokens.shape[1]
    max_len = model.max_sequence_length

    for _ in range(max_new_tokens):

        # Keep within the model's max sequence length
        context = input_tokens[:, -max_len:]

        logits = model(context)
        next_token_logits = logits[:, -1, :]

        if temperature and temperature > 0:
            scaled_logits = next_token_logits / temperature

            if top_k is not None:
                top_values, _ = torch.topk(scaled_logits, top_k)
                threshold = top_values[:, -1].unsqueeze(-1)
                scaled_logits = scaled_logits.masked_fill(
                    scaled_logits < threshold, float("-inf")
                )

            probabilities = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1)

        input_tokens = torch.cat(
            [input_tokens, next_token.unsqueeze(0)], dim=1
        )

    generated_ids = input_tokens[0].tolist()
    full_text = tokenizer.decode(generated_ids)
    completion_ids = generated_ids[prompt_length:]
    completion_text = tokenizer.decode(completion_ids)

    return {
        "full_text": full_text,
        "completion": completion_text,
        "prompt_tokens": prompt_length,
        "completion_tokens": len(completion_ids),
    }


@torch.no_grad()
def generate_with_internals(
    model: TinyLLM,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_k_logits: int = 8,
    device: torch.device | None = None,
) -> dict:
    """Same autoregressive loop as ``generate``, but at every step it
    also records what the model's internals looked like while
    producing that token: per-layer / per-head attention weights,
    an embedding fingerprint per context token, and the top candidate
    tokens with their probabilities.

    This is intentionally a separate function rather than a flag on
    ``generate`` — capturing internals is noticeably more expensive
    (Python-side list building every step) and callers that just want
    text (training scripts, the plain chat endpoint) shouldn't pay for
    it.
    """

    device = device or next(model.parameters()).device

    prompt_ids = tokenizer.encode(prompt)
    input_tokens = torch.tensor(
        [prompt_ids], dtype=torch.long, device=device
    )

    prompt_length = input_tokens.shape[1]
    max_len = model.max_sequence_length

    steps = []

    for step_index in range(max_new_tokens):

        context = input_tokens[:, -max_len:]
        context_ids = context[0].tolist()
        context_tokens = [
            tokenizer.itos.get(t, "<UNK>") for t in context_ids
        ]

        internals = model(context, return_internals=True)

        logits = internals["logits"]
        next_token_logits = logits[:, -1, :]

        # Embedding fingerprint per context token: a small, fixed-size
        # slice of the embedding vector (not the full 128-dim vector,
        # which is more than a UI needs) plus its L2 norm, so the
        # frontend can draw *something* concrete for "tokens become
        # vectors" without shipping megabytes of floats.
        embedding_output = internals["embedding_output"][0]  # [T, D]
        embedding_fingerprints = [
            {
                "preview": embedding_output[t, :6].tolist(),
                "norm": embedding_output[t].norm().item(),
            }
            for t in range(embedding_output.shape[0])
        ]

        layers = []
        for layer_index, attention_weights in enumerate(
            internals["layer_attention_weights"]
        ):
            # attention_weights: [B, H, T, T] -> batch 0 -> per head T x T
            heads = attention_weights[0].tolist()
            layers.append({
                "layer_index": layer_index,
                "heads": heads,
            })

        # Softmax over the *unscaled* logits (temperature 0 / greedy is
        # the project's default) so the probabilities shown always
        # reflect what argmax is actually choosing between; when a
        # temperature is set we show the scaled distribution instead,
        # matching what sampling actually draws from.
        if temperature and temperature > 0:
            scaled_logits = next_token_logits / temperature

            if top_k is not None:
                top_values, _ = torch.topk(scaled_logits, top_k)
                threshold = top_values[:, -1].unsqueeze(-1)
                scaled_logits = scaled_logits.masked_fill(
                    scaled_logits < threshold, float("-inf")
                )

            probabilities = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
        else:
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(next_token_logits, dim=-1)

        k = min(top_k_logits, probabilities.shape[-1])
        top_probs, top_ids = torch.topk(probabilities[0], k)
        top_candidates = [
            {
                "token": tokenizer.itos.get(tid, "<UNK>"),
                "probability": prob,
            }
            for tid, prob in zip(top_ids.tolist(), top_probs.tolist())
        ]

        sampled_id = next_token.item()
        sampled_token = tokenizer.itos.get(sampled_id, "<UNK>")

        steps.append({
            "step_index": step_index,
            "context_tokens": context_tokens,
            "embeddings": embedding_fingerprints,
            "layers": layers,
            "top_candidates": top_candidates,
            "sampled_token": sampled_token,
        })

        input_tokens = torch.cat(
            [input_tokens, next_token.unsqueeze(0)], dim=1
        )

    generated_ids = input_tokens[0].tolist()
    full_text = tokenizer.decode(generated_ids)
    completion_ids = generated_ids[prompt_length:]
    completion_text = tokenizer.decode(completion_ids)

    return {
        "full_text": full_text,
        "completion": completion_text,
        "prompt_tokens": prompt_length,
        "completion_tokens": len(completion_ids),
        "num_layers": len(model.transformer_blocks),
        "num_heads": model.transformer_blocks[0].attention.num_heads,
        "steps": steps,
    }


def generate_chat_reply_with_internals(
    model: TinyLLM,
    tokenizer: SimpleTokenizer,
    user_message: str,
    max_new_tokens: int = 20,
    temperature: float = 0.0,
    top_k: int | None = None,
    device: torch.device | None = None,
) -> dict:
    """``generate_chat_reply``'s counterpart for the visualizer: wraps
    the message in the same "User: ... Assistant:" template and
    returns the reply plus the full step-by-step internals trace."""

    prompt = f"User: {user_message} Assistant:"

    result = generate_with_internals(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )

    reply = result["full_text"].split("Assistant:", maxsplit=1)[-1].strip()
    result["reply"] = reply
    return result


def generate_chat_reply(
    model: TinyLLM,
    tokenizer: SimpleTokenizer,
    user_message: str,
    max_new_tokens: int = 20,
    temperature: float = 0.0,
    top_k: int | None = None,
    device: torch.device | None = None,
) -> dict:
    """Wrap a raw user message in the "User: ... Assistant:" template
    the SFT checkpoint was trained on, and return only the assistant's
    reply (everything after the first "Assistant:")."""

    prompt = f"User: {user_message} Assistant:"

    result = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )

    reply = result["full_text"].split("Assistant:", maxsplit=1)[-1].strip()
    result["reply"] = reply
    return result
