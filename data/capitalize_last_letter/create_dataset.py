import json
from datasets import Dataset
with open("capitalize_last_letter.json", "r") as file:
    data = json.load(file)

test_proportion=0.2
seed=42

dataset = {
    "input": [],
    "target": []
}
for item in data:
    dataset["input"].append(item["input"])
    dataset["target"].append(item["output"])

hf_dataset = Dataset.from_dict(dataset).train_test_split(test_size=test_proportion, seed=seed)
save_path = "./capitalize_last_letter"
hf_dataset.save_to_disk(save_path)