import torch

def collate_fn(batch, num_soft_prompt_tokens, text_column, label_column, tokenizer):
    contexts, targets = zip(*[(item[text_column], item[label_column]) for item in batch])
    inputs = [f"Input: {x}\nLabel: " for x in contexts]

    special_token = "<special-token-we-will-never-encounter-in-a-dataset>"
    tokenizer.add_tokens([special_token])

    soft_prompt_tokens = [f"<soft-prompt-token-{i}>" for i in range(num_soft_prompt_tokens)]
    soft_prompt_token_ids = [tokenizer.convert_tokens_to_ids([soft_prompt_token])[0] for soft_prompt_token in soft_prompt_tokens]

    tokenized_inputs = [tokenizer(inp) for inp in inputs]
    tokenized_labels = [tokenizer(special_token + target, add_special_tokens=False) for target in targets]
    assert len(tokenized_labels) == len(tokenized_inputs)
    tokenized_inputs_with_labels = {}
    tokenized_inputs_with_labels["input_ids"] = [soft_prompt_token_ids + tok_inp["input_ids"] + tok_target["input_ids"][1:] for tok_inp, tok_target in zip(tokenized_inputs, tokenized_labels)]
    tokenized_inputs_with_labels["attention_mask"] = [[1] * len(soft_prompt_token_ids) + tok_inp["attention_mask"] + tok_target["attention_mask"][1:] for tok_inp, tok_target in zip(tokenized_inputs, tokenized_labels)]
    tokenized_inputs_with_labels["labels"] = [[-100] * (len(tok_inp["input_ids"]) + len(soft_prompt_token_ids)) + tok_target["input_ids"][1:] for tok_inp, tok_target in zip(tokenized_inputs, tokenized_labels)]
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

    tokenized_only_inputs = {}
    tokenized_only_inputs["input_ids"] = [soft_prompt_token_ids + tok_inp["input_ids"] for tok_inp in tokenized_inputs]
    tokenized_only_inputs["attention_mask"] = [[1] * len(soft_prompt_token_ids) + tok_inp["attention_mask"] for tok_inp in tokenized_inputs]
    max_len_tokenized_inputs = max([len(seq) for seq in tokenized_only_inputs["input_ids"]])
    tokenized_only_inputs["input_ids"] = [[tokenizer.pad_token_id] * (max_len_tokenized_inputs - len(seq)) + seq
                        for seq in tokenized_only_inputs["input_ids"]]
    tokenized_only_inputs["attention_mask"] = [[0] * (max_len_tokenized_inputs - len(seq)) + seq
                        for seq in tokenized_only_inputs["attention_mask"]]
    
    for tp in tokenized_only_inputs:
        tokenized_only_inputs[tp] = torch.tensor(tokenized_only_inputs[tp])[..., :max_seq_len]

    assert tokenized_inputs_with_labels["input_ids"].shape[-1] <= 1024 and tokenized_inputs_with_labels["attention_mask"].shape[-1] <= 1024 and tokenized_inputs_with_labels["labels"].shape[-1] <= 1024
    return tokenized_inputs_with_labels, tokenized_only_inputs