class SimpleTokenizer:
    def __init__(self, text):
        # Build vocabulary from unique words
        words = sorted(set(text.split()))

        self.stoi = {
            "<PAD>": 0,
            "<UNK>": 1
        }

        for word in words:
            if word not in self.stoi:
                self.stoi[word] = len(self.stoi)

        self.itos = {i: word for word, i in self.stoi.items()}

    def encode(self, text):
        return [
            self.stoi.get(word, self.stoi["<UNK>"])
            for word in text.split()
        ]

    def decode(self, token_ids):
        return " ".join(
            self.itos.get(token_id, "<UNK>")
            for token_id in token_ids
        )