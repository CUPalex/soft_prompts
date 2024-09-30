import torch
import os
import sys
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.collate_fn_with_pretrained_soft_prompt import collate_fn
from functools import partial
import wandb
import numpy as np

def get_task_import_dataset_func(dataset_name):
    if dataset_name == "sst2":
        from data.sst2.load_dataset import get_train_test_dataset
    elif dataset_name == "country_capital":
        from data.country_capital.load_dataset import get_train_test_dataset
    elif dataset_name == "landmark_country":
        from data.landmark_country.load_dataset import get_train_test_dataset
    elif dataset_name == "singular_plural":
        from data.singular_plural.load_dataset import get_train_test_dataset
    elif dataset_name == "present_past":
        from data.present_past.load_dataset import get_train_test_dataset
    elif dataset_name == "capitalize_first_letter":
        from data.capitalize_first_letter.load_dataset import get_train_test_dataset
    elif dataset_name == "capitalize_last_letter":
        from data.capitalize_last_letter.load_dataset import get_train_test_dataset
    elif dataset_name == "lowercase_first_letter":
        from data.lowercase_first_letter.load_dataset import get_train_test_dataset
    elif dataset_name == "lowercase_last_letter":
        from data.lowercase_last_letter.load_dataset import get_train_test_dataset
    elif dataset_name == "country_currency":
        from data.country_currency.load_dataset import get_train_test_dataset
    elif dataset_name == "ag_news":
        from data.ag_news.load_dataset import get_train_test_dataset
    elif dataset_name == "yelp_polarity":
        from data.yelp_polarity.load_dataset import get_train_test_dataset
    elif dataset_name == "snli":
        from data.snli.load_dataset import get_train_test_dataset
    elif dataset_name == "cola":
        from data.cola.load_dataset import get_train_test_dataset
    elif dataset_name == "wnli":
        from data.wnli.load_dataset import get_train_test_dataset
    elif dataset_name == "rte":
        from data.rte.load_dataset import get_train_test_dataset
    elif dataset_name == "financial_phrasebank":
        from data.financial_phrasebank.load_dataset import get_train_test_dataset
    else:
        raise NotImplementedError()
    return get_train_test_dataset

def run():
    device = "cuda:0"
    model_name_or_path = "openai-community/gpt2-xl"
    simple_words = ['cat', 'dog', 'bird', 'book', 'house', 'chair', 'table', 'pen', 
                    'paper', 'water', 'food', 'fish', 'hand', 'foot', 'ball', 'cup', 'phone',
                    'light', 'road', 'bag', 'clock', 'mouse', 'key', 'rain', 'wind', 'river', 'apple',
                    'grass', 'leaf', 'wall', 'window', 'box', 'flower', 'star', 'cake'][:10]
    tasks = ["country_capital", "landmark_country", "present_past", "singular_plural", "capitalize_first_letter", "lowercase_first_letter"]
    num_virtual_tokens = 4
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model = model.to(device)
    model.eval()

    num_soft_prompt_tokens = 1
    batch_size = 8

    new_tokens = [f"<soft-prompt-token-{i}>" for i in range(num_soft_prompt_tokens)]
    assert all([token not in tokenizer.vocab.keys() for token in new_tokens])
    tokenizer.add_tokens(new_tokens)
    prev_size = model.transformer.wte.weight.shape[0]
    prev_emb = model.transformer.wte.weight[0]
    model.resize_token_embeddings(len(tokenizer))
    assert prev_size == model.transformer.wte.weight.shape[0] - len(new_tokens)
    assert (prev_emb == model.transformer.wte.weight[0]).all()

    run = wandb.init(
        project="soft-prompts-run-with-average",
        name = f"num_soft_prompt_tokens-{num_soft_prompt_tokens}",
        config=dict(
            batch_size = batch_size,
            model=model_name_or_path,
            type_of_emb="average",
            num_soft_prompt_tokens=num_soft_prompt_tokens,
        ),
    )

    for task in tasks:
        get_train_test_dataset = get_task_import_dataset_func(task)
        dataset, text_column, label_column = get_train_test_dataset()
        eval_dataloader = DataLoader(
            dataset["test"],
            collate_fn=partial(collate_fn, num_soft_prompt_tokens=len(new_tokens), text_column=text_column, label_column=label_column, tokenizer=tokenizer),
            batch_size=batch_size, pin_memory=True
        )
        avg_embedding = np.load(f"embeddings/average_emb_of_tasks/embedding-{task}.npy")
        with torch.no_grad():
            for i in range(num_soft_prompt_tokens):
                model.transformer.wte.weight[tokenizer.convert_tokens_to_ids([f"<soft-prompt-token-{i}>"])[0]] = torch.tensor(avg_embedding,
                                                                        device=model.transformer.wte.weight.device,
                                                                        dtype=model.transformer.wte.weight.dtype)

        save_every = 50
        correct = 0
        eval_loss = 0
        loss_over_batch = 0
        continuations = wandb.Table(columns=["full output", "label", "continuation"])
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch_loss = {k: v.to(device) for k, v in batch[0].items()}
            assert batch_loss["input_ids"].shape[-1] <= 1024 and batch_loss["attention_mask"].shape[-1] <= 1024 and batch_loss["labels"].shape[-1] <= 1024
            with torch.no_grad():
                outputs = model(**batch_loss)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            loss_over_batch += loss.detach().float()
            
            batch_generate = {k: v.to(device) for k, v in batch[1].items()}
            out = model.generate(**batch_generate, max_new_tokens=5)
            for i in range(batch_generate["input_ids"].shape[0]):
                continuation = tokenizer.decode(out[i][batch_generate["input_ids"][i].shape[0]:])
                eval_correct_labels = tokenizer.decode(batch_loss["labels"][i][batch_loss["labels"][i] != -100], skip_special_tokens=True)
                if step < 3:
                    continuations.add_data(tokenizer.decode(out[i]), eval_correct_labels, continuation)
                correct += int(continuation.strip().startswith(eval_correct_labels))
            if step % save_every == 0 and step > 0:
                wandb.log({f"{task}/eval/loss": loss_over_batch / save_every, f"{task}/eval/cur_num_correct": correct})
                loss_over_batch = 0

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        eval_accuracy = correct / len(dataset["test"])
        wandb.log({
                f"{task}/eval_ppl": eval_ppl,
                f"{task}/eval_accuracy": eval_accuracy,
                f"{task}/eval_epoch_loss": eval_epoch_loss})
        run.log({f"{task}/continuations" : continuations})
    run.finish()

if __name__ == "__main__":
    run()