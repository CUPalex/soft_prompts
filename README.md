# :cloud: Are soft prompts actually not interpretable?

## :cyclone: About
In this work we explored the geometry of soft prompts trained with prompt tuning for different tasks and with different initializations. We located the trained tokens as well as some words from the vocabulary of the model in the embedding space. We found that while the initialization is likely to play a more important role in the location of the trained token, than the task it is trained for, the location of the prompt token is still highly dependent on the target task.

The results of the work are presented in the ```robust_nlp_paper.pdf``` file in the root of the repository.

## :cyclone: Usage

The main script is ```train_prompts.py``` - it utilizes transformers library to train prompts for the datasets located in ```data``` directory and logs to wandb.

## :cyclone: Files structure: important files

```
.
├── data
│   └── dataset_name
│   │   └── load_dataset.py # contains a function get_train_test_dataset() that returs a dataset in a unified format
│   ├── collate_fn.py # function with preprocessing logic for training prompts
│   └── collate_fn_with_pretrained_soft_prompt.py # function with preprocessing logic if running a model with already pre-trained prompts
├── robust_nlp_paper.pdf
└── train_prompts.py # script for training prompts
```

## :cyclone: Licence
MIT
