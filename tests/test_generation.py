import torch

from tiny_llm.model import TinyLLM
from tiny_llm.tokenizer import SimpleTokenizer
from tiny_llm.generation import generate, generate_chat_reply


def make_model_and_tokenizer():
    text = "User: What is machine learning? Assistant: A method for computers to learn."
    tokenizer = SimpleTokenizer(text)

    model = TinyLLM(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=16,
        num_heads=2,
        ff_hidden_dim=32,
        num_layers=2,
        max_sequence_length=32,
    )
    model.eval()
    return model, tokenizer


def test_greedy_generation_appends_requested_number_of_tokens():
    model, tokenizer = make_model_and_tokenizer()

    result = generate(model, tokenizer, "User: Hi", max_new_tokens=5, device=torch.device("cpu"))

    assert result["completion_tokens"] == 5
    assert result["prompt_tokens"] == len(tokenizer.encode("User: Hi"))


def test_greedy_generation_is_deterministic():
    model, tokenizer = make_model_and_tokenizer()

    result_a = generate(model, tokenizer, "User: Hi", max_new_tokens=5, device=torch.device("cpu"))
    result_b = generate(model, tokenizer, "User: Hi", max_new_tokens=5, device=torch.device("cpu"))

    assert result_a["completion"] == result_b["completion"]


def test_sampling_respects_temperature_without_crashing():
    model, tokenizer = make_model_and_tokenizer()

    result = generate(
        model, tokenizer, "User: Hi",
        max_new_tokens=5, temperature=0.8, top_k=5, device=torch.device("cpu"),
    )

    assert result["completion_tokens"] == 5


def test_chat_reply_strips_the_prompt_template():
    model, tokenizer = make_model_and_tokenizer()

    result = generate_chat_reply(model, tokenizer, "Hi", max_new_tokens=5, device=torch.device("cpu"))

    assert "reply" in result
    assert "Assistant:" not in result["reply"]
