from datasets import load_dataset

MAX_ITEMS_IN_TRAIN = 10000

def get_train_test_dataset():
    dataset = load_dataset("fancyzhx/ag_news")
    text_column = "text"
    original_label_column = "label"
    label_column = "text_label"

    classes = [cl.replace("_", " ") for cl in dataset["train"].features[original_label_column].names]
    dataset = dataset.map(
        lambda x: {label_column: [classes[label] for label in x[original_label_column]]},
        batched=True,
        num_proc=1,
    )
    dataset["train"] = dataset["train"].train_test_split(train_size=MAX_ITEMS_IN_TRAIN,
                                                    stratify_by_column=original_label_column)["train"]
    return dataset, text_column, label_column