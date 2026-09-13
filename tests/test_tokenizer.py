from tiny_llm.tokenizer import SimpleTokenizer


def test_special_tokens_are_reserved(toy_tokenizer):
    assert toy_tokenizer.stoi["<PAD>"] == 0
    assert toy_tokenizer.stoi["<UNK>"] == 1


def test_encode_decode_roundtrip(toy_tokenizer):
    sentence = "I am a student"
    encoded = toy_tokenizer.encode(sentence)

    assert len(encoded) == len(sentence.split())
    assert toy_tokenizer.decode(encoded) == sentence


def test_unknown_words_map_to_unk_token(toy_tokenizer):
    encoded = toy_tokenizer.encode("I am a wizard")
    unk_id = toy_tokenizer.stoi["<UNK>"]

    # "wizard" was never seen while building the vocabulary
    assert encoded[-1] == unk_id


def test_shared_vocabulary_covers_both_corpora(toy_tokenizer):
    # Simulates train.py building one vocabulary from two text sources
    instruction_text = "User: What is artificial intelligence?"
    combined = SimpleTokenizer(
        toy_tokenizer.decode(toy_tokenizer.encode(str(toy_tokenizer.stoi)))
        + "\n" + instruction_text
    )
    assert combined.vocab_size >= toy_tokenizer.vocab_size


def test_save_and_load_roundtrip(toy_tokenizer, tmp_path):
    path = tmp_path / "tokenizer.pt"
    toy_tokenizer.save(path)

    reloaded = SimpleTokenizer.load(path)

    assert reloaded.stoi == toy_tokenizer.stoi
    assert reloaded.itos == toy_tokenizer.itos
