import torch

def collate_fn(batch, text_column, label_column, tokenizer):
    contexts, targets = zip(*[(item[text_column], item[label_column]) for item in batch])
    inputs = [f"Input: {x}\nLabel: " for x in contexts]

    special_token = "<special-token-we-will-never-encounter-in-a-dataset>"
    tokenizer.add_tokens([special_token])

    tokenized_inputs = [tokenizer(inp) for inp in inputs]
    tokenized_labels = [tokenizer(special_token + target, add_special_tokens=False) for target in targets]
    assert len(tokenized_labels) == len(tokenized_inputs)
    tokenized_inputs_with_labels = {}
    tokenized_inputs_with_labels["input_ids"] = [tok_inp["input_ids"] + tok_target["input_ids"][1:] for tok_inp, tok_target in zip(tokenized_inputs, tokenized_labels)]
    tokenized_inputs_with_labels["attention_mask"] = [tok_inp["attention_mask"] + tok_target["attention_mask"][1:] for tok_inp, tok_target in zip(tokenized_inputs, tokenized_labels)]
    tokenized_inputs_with_labels["labels"] = [[-100] * len(tok_inp["input_ids"]) + tok_target["input_ids"][1:] for tok_inp, tok_target in zip(tokenized_inputs, tokenized_labels)]
    max_len_tokenized_inputs_with_labels = max([len(seq) for seq in tokenized_inputs_with_labels["input_ids"]])
    tokenized_inputs_with_labels["input_ids"] = [[tokenizer.pad_token_id] * (max_len_tokenized_inputs_with_labels - len(seq)) + seq
                        for seq in tokenized_inputs_with_labels["input_ids"]]
    tokenized_inputs_with_labels["attention_mask"] = [[0] * (max_len_tokenized_inputs_with_labels - len(seq)) + seq
                        for seq in tokenized_inputs_with_labels["attention_mask"]]
    tokenized_inputs_with_labels["labels"] = [[-100] * (max_len_tokenized_inputs_with_labels - len(seq)) + seq
                        for seq in tokenized_inputs_with_labels["labels"]]

    max_seq_len = 1024

    for tp in tokenized_inputs_with_labels:
        tokenized_inputs_with_labels[tp] = torch.tensor(tokenized_inputs_with_labels[tp])[..., :max_seq_len]

    tokenizer.padding_side = "left"
    tokenized_only_inputs = tokenizer(inputs, return_tensors="pt", padding=True)
    tokenized_only_inputs["input_ids"] = tokenized_only_inputs["input_ids"][..., :max_seq_len]
    tokenized_only_inputs["attention_mask"] = tokenized_only_inputs["attention_mask"][..., :max_seq_len]

    assert tokenized_inputs_with_labels["input_ids"].shape[-1] <= 1024 and tokenized_inputs_with_labels["attention_mask"].shape[-1] <= 1024 and tokenized_inputs_with_labels["labels"].shape[-1] <= 1024
    return tokenized_inputs_with_labels, tokenized_only_inputs