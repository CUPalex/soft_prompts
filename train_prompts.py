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
from data.collate_fn import collate_fn
from functools import partial
import wandb
import numpy as np

def run(dataset_name):
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

    device = "cuda:0"
    model_name_or_path = "openai-community/gpt2-xl"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    simple_words = ['cat', 'dog', 'bird', 'book', 'house', 'chair', 'table', 'pen', 
                    'paper', 'water', 'food', 'fish', 'hand', 'foot', 'ball', 'cup', 'phone',
                    'light', 'road', 'bag', 'clock', 'mouse', 'key', 'rain', 'wind', 'river', 'apple',
                    'grass', 'leaf', 'wall', 'window', 'box', 'flower', 'star', 'cake']

    num_virtual_tokens = 4
    num_epochs = 50
    batch_size = 8

    dataset, text_column, label_column = get_train_test_dataset()
    train_dataloader = DataLoader(
        dataset["train"], shuffle=True,
        collate_fn=partial(collate_fn, text_column=text_column, label_column=label_column, tokenizer=tokenizer),
        batch_size=batch_size, pin_memory=True
    )

    eval_dataloader = DataLoader(
        dataset["test"],
        collate_fn=partial(collate_fn, text_column=text_column, label_column=label_column, tokenizer=tokenizer),
        batch_size=batch_size, pin_memory=True
    )

    for word in simple_words[:10]:
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=num_virtual_tokens,
            prompt_tuning_init_text="".join([word] * num_virtual_tokens),
            tokenizer_name_or_path=model_name_or_path,
        )

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        model = get_peft_model(model, peft_config)
        print(model.print_trainable_parameters())
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters())
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(train_dataloader) * num_epochs),
        )

        run = wandb.init(
            project="soft-prompts-10",
            name = f"{dataset_name}-{word}",
            config=dict(
                dataset_name = dataset_name,
                word=word,
                num_virtual_tokens=num_virtual_tokens,
                num_epochs=num_epochs,
                model=model_name_or_path,
                batch_size = batch_size,
            ),
        )
        save_every = 50
        continuations = wandb.Table(columns=["full output", "label", "continuation"])
        embeddings = []
        for epoch in range(num_epochs):
            model.train()
            total_loss, loss_over_batch = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                batch_loss = {k: v.to(device) for k, v in batch[0].items()}
                outputs = model(**batch_loss)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss_over_batch += loss.detach().float()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if step % save_every == 0 and step > 0:
                    wandb.log({"train/loss": loss_over_batch / save_every})
                    loss_over_batch = 0

            model.eval()
            correct = 0
            eval_loss, loss_over_batch = 0, 0
            for step, batch in enumerate(tqdm(eval_dataloader)):
                batch_loss = {k: v.to(device) for k, v in batch[0].items()}
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
                    wandb.log({"eval/loss": loss_over_batch / save_every, "eval/cur_num_correct": correct})
                    loss_over_batch = 0

            embeddings.append(model.prompt_encoder.default.embedding.weight.detach().cpu().numpy())
            eval_epoch_loss = eval_loss / len(eval_dataloader)
            eval_ppl = torch.exp(eval_epoch_loss)
            eval_accuracy = correct / len(dataset["test"])
            wandb.log({
                    "epoch/epoch": epoch,
                    "epoch/eval_ppl": eval_ppl,
                    "epoch/eval_accuracy": eval_accuracy,
                    "epoch/eval_epoch_loss": eval_epoch_loss})
        embeddings = np.stack(embeddings)
        np.save(f"/proj/inductive-bias/llama/soft_prompts/embeddings/{dataset_name}-{word}", embeddings)
        run.log({"artifacts/continuations" : continuations})
        run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", "-d")
    args = parser.parse_args()
    run(args.dataset)