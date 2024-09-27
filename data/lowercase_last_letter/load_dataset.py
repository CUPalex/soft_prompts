from datasets import load_from_disk

def get_train_test_dataset():
    dataset_name = "/proj/inductive-bias/llama/soft_prompts/data/lowercase_last_letter/lowercase_last_letter"
    text_column = "input"
    label_column = "target"
    dataset = load_from_disk(dataset_name)

    return dataset, text_column, label_column