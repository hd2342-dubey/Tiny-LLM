from src.tokenizer.tokenizer import SimpleTokenizer

text = """
I am a student
I am learning artificial intelligence
I love deep learning
"""

tokenizer = SimpleTokenizer(text)

sentence = "I am a student"

encoded = tokenizer.encode(sentence)

print("Vocabulary:")
print(tokenizer.stoi)

print("\nEncoded:")
print(encoded)

print("\nDecoded:")
print(tokenizer.decode(encoded))