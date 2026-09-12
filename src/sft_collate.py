import torch


def sft_collate_fn(batch):

    input_sequences = [item[0] for item in batch]
    target_sequences = [item[1] for item in batch]

    max_length = max(
        len(sequence)
        for sequence in input_sequences
    )

    padded_inputs = []
    padded_targets = []

    for input_tokens, target_tokens in zip(
        input_sequences,
        target_sequences
    ):

        padding_length = max_length - len(input_tokens)

        # ----------------------------------
        # Pad inputs with <PAD> = 0
        # ----------------------------------

        padded_input = torch.cat(
            [
                input_tokens,
                torch.zeros(
                    padding_length,
                    dtype=torch.long
                )
            ]
        )

        # ----------------------------------
        # Pad targets with -100
        # ----------------------------------
        
        padded_target = torch.cat(
            [
                target_tokens,
                torch.full(
                    (padding_length,),
                    -100,
                    dtype=torch.long
                )
            ]
        )

        padded_inputs.append(padded_input)
        padded_targets.append(padded_target)

    return (
        torch.stack(padded_inputs),
        torch.stack(padded_targets)
    )