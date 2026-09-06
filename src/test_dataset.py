from dataset import LanguageModelDataset

token_ids = [10, 20, 30, 40, 50, 60, 70]

sequence_length = 4

dataset = LanguageModelDataset(
    token_ids=token_ids,
    sequence_length=sequence_length
)

print("Dataset length:", len(dataset))

for i in range(len(dataset)):

    input_tokens, target_tokens = dataset[i]

    print(f"\nExample {i + 1}")
    print("Input :", input_tokens.tolist())
    print("Target:", target_tokens.tolist())