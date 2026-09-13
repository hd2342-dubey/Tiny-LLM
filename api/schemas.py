from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["sft", "pretrained"]


class GenerateRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=300)
    mode: Mode = "sft"
    max_new_tokens: int = Field(default=25, ge=1, le=80)
    temperature: float = Field(default=0.0, ge=0.0, le=1.5)
    top_k: int | None = Field(default=None, ge=1, le=50)


class GenerateResponse(BaseModel):
    reply: str
    mode: Mode
    prompt_tokens: int
    completion_tokens: int
    prompt_token_preview: list[str]
    contained_unknown_words: bool


class ModelInfo(BaseModel):
    vocab_size: int
    embedding_dim: int
    num_heads: int
    head_dim: int
    num_layers: int
    ff_hidden_dim: int
    max_sequence_length: int
    num_parameters: int
    device: str
    available_modes: list[Mode]


class HealthResponse(BaseModel):
    status: str
    device: str
    modes_loaded: list[Mode]


class GenerateInternalsRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=300)
    mode: Mode = "sft"
    # Capped lower than /api/generate: each step ships a full
    # attention matrix per layer/head, so payload size grows with
    # both prompt length and step count.
    max_new_tokens: int = Field(default=16, ge=1, le=40)
    temperature: float = Field(default=0.0, ge=0.0, le=1.5)
    top_k: int | None = Field(default=None, ge=1, le=50)


class TokenEmbedding(BaseModel):
    preview: list[float]
    norm: float


class LayerAttention(BaseModel):
    layer_index: int
    # heads[h][query_pos][key_pos] = attention weight
    heads: list[list[list[float]]]


class TopCandidate(BaseModel):
    token: str
    probability: float


class GenerationStep(BaseModel):
    step_index: int
    context_tokens: list[str]
    embeddings: list[TokenEmbedding]
    layers: list[LayerAttention]
    top_candidates: list[TopCandidate]
    sampled_token: str


class GenerateInternalsResponse(BaseModel):
    reply: str
    mode: Mode
    prompt_tokens: int
    completion_tokens: int
    num_layers: int
    num_heads: int
    steps: list[GenerationStep]
