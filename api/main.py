"""
FastAPI backend for the Tiny LLM chat demo.

Loads the tokenizer and both checkpoints (pretrained + instruction-
tuned) once at startup, then serves:

  GET  /api/health        - liveness + which checkpoints loaded
  GET  /api/model-info    - architecture stats for the "internals" panel
  POST /api/generate      - run a chat message through the model
  GET  /                  - the static chat UI (web/index.html)

Run with:
    uvicorn api.main:app --reload
or:
    tiny-llm-serve
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from tiny_llm.checkpoint_utils import CHECKPOINTS, load_model, load_tokenizer
from tiny_llm.config import DEFAULT_CONFIG, get_device
from tiny_llm.generation import generate_chat_reply, generate_chat_reply_with_internals

from api.schemas import (
    GenerateInternalsRequest,
    GenerateInternalsResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfo,
)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

state: dict = {"tokenizer": None, "models": {}, "device": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = get_device()
    tokenizer = load_tokenizer()

    models = {}
    for mode, path in CHECKPOINTS.items():
        if path.exists():
            models[mode] = load_model(mode, tokenizer, device=device)

    state["tokenizer"] = tokenizer
    state["models"] = models
    state["device"] = device

    print(f"Tiny LLM API ready — device={device}, modes loaded={list(models)}")
    yield
    state.clear()


app = FastAPI(title="Tiny LLM API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_model(mode: str):
    model = state["models"].get(mode)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No '{mode}' checkpoint is loaded. Train it first with "
                f"`tiny-llm-train`{' and `tiny-llm-train-sft`' if mode == 'sft' else ''}."
            ),
        )
    return model


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if state["models"] else "no checkpoints loaded",
        device=str(state["device"]),
        modes_loaded=list(state["models"].keys()),
    )


@app.get("/api/model-info", response_model=ModelInfo)
def model_info():
    tokenizer = state["tokenizer"]
    # Any loaded model has the same architecture; prefer sft, fall back to pretrained.
    model = state["models"].get("sft") or state["models"].get("pretrained")

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    config = DEFAULT_CONFIG

    return ModelInfo(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        head_dim=config.embedding_dim // config.num_heads,
        num_layers=config.num_layers,
        ff_hidden_dim=config.ff_hidden_dim,
        max_sequence_length=config.max_sequence_length,
        num_parameters=model.num_parameters(),
        device=str(state["device"]),
        available_modes=list(state["models"].keys()),
    )


@app.post("/api/generate", response_model=GenerateResponse)
def generate_endpoint(request: GenerateRequest):
    tokenizer = state["tokenizer"]
    model = _require_model(request.mode)

    result = generate_chat_reply(
        model,
        tokenizer,
        request.message,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        device=state["device"],
    )

    prompt_ids = tokenizer.encode(f"User: {request.message} Assistant:")
    prompt_preview = [tokenizer.itos.get(t, "<UNK>") for t in prompt_ids]
    unk_id = tokenizer.stoi["<UNK>"]

    return GenerateResponse(
        reply=result["reply"] or "(the model generated an empty response)",
        mode=request.mode,
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        prompt_token_preview=prompt_preview,
        contained_unknown_words=unk_id in prompt_ids,
    )


@app.post("/api/generate-with-internals", response_model=GenerateInternalsResponse)
def generate_with_internals_endpoint(request: GenerateInternalsRequest):
    tokenizer = state["tokenizer"]
    model = _require_model(request.mode)

    result = generate_chat_reply_with_internals(
        model,
        tokenizer,
        request.message,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        device=state["device"],
    )

    return GenerateInternalsResponse(
        reply=result["reply"] or "(the model generated an empty response)",
        mode=request.mode,
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        num_layers=result["num_layers"],
        num_heads=result["num_heads"],
        steps=result["steps"],
    )


# Serve the static chat UI at "/"
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
