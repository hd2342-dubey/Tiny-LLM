from src.tokenizer.tokenizer import SimpleTokenizer


# ==========================================
# Load both datasets
# ==========================================

with open("data/train.txt", "r", encoding="utf-8") as f:
    pretraining_text = f.read()

with open("data/instructions.txt", "r", encoding="utf-8") as f:
    instruction_text = f.read()


# ==========================================
# Combine text for vocabulary creation
# ==========================================

combined_text = pretraining_text + "\n" + instruction_text


# ==========================================
# Create tokenizer
# ==========================================

tokenizer = SimpleTokenizer(combined_text)


# ==========================================
# Display vocabulary information
# ==========================================

print("Vocabulary size:", len(tokenizer.stoi))

print("\nSpecial tokens:")
print("<PAD>:", tokenizer.stoi["<PAD>"])
print("<UNK>:", tokenizer.stoi["<UNK>"])


# ==========================================
# Check instruction tokens
# ==========================================

test_sentence = "User: What is artificial intelligence?"

encoded = tokenizer.encode(test_sentence)

print("\nTest sentence:")
print(test_sentence)

print("\nEncoded:")
print(encoded)

print("\nDecoded:")
print(tokenizer.decode(encoded))


# ==========================================
# Check for unknown tokens
# ==========================================

unknown_id = tokenizer.stoi["<UNK>"]

unknown_count = encoded.count(unknown_id)

print("\nUnknown tokens:", unknown_count)