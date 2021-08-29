# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import os
import math
import datetime

from pathlib import Path
from torch.utils import data
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizerFast
from transformers import LineByLineTextDataset
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from tokenizers.implementations import ByteLevelBPETokenizer

from dataset import CoLMDataset

def train_tokenizer(
    data_dir: str='./data', 
    vocab_size: int=52000, 
    min_frequency: int=2, 
    output_dir: str='./models', 
    prefix: str='coberta-mini',
    **kwargs
):
    paths = [str(x) for x in Path(data_dir).glob('*.txt')]

    # TODO: Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # TODO: Customize training
    tokenizer.train(
        files=paths,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_token=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>"
        ]
    )

    # TODO: Save files
    tokenizer.save_model(output_dir, prefix)

def train_language_model(
    data_dir: str='./data',
    pretrained_path: str='./models/coberta-mini',
    vocab_size: int=52000,
    max_position_embeddings: int=512,
    num_attention_heads: int=12,
    num_hidden_layers: int=6,
    type_vocab_size: int=1,
    max_len: int=256,
    block_size: int=128,
    output_dir: str='./models/coberta-mini',
    num_train_epochs: int=40,
    per_gpu_train_batch_size: int=64,
    save_steps: int=10_000,
    save_total_limit: int=2,
    prediction_loss_only: bool=True,
    **kwargs
):
    config = RobertaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        type_vocab_size=type_vocab_size,
    )

    tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_path, max_len=max_len)

    model = RobertaForMaskedLM(config=config)
    print(f"🔥 The numbers of parameters: {model.num_parameters()}")

    train_dataset = CoLMDataset(
        root=data_dir, 
        mode='train',
        tokenizer=tokenizer
    )
    eval_dataset = CoLMDataset(
        root=data_dir,
        mode='eval',
        tokenizer=tokenizer
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_gpu_train_batch_size=per_gpu_train_batch_size,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        prediction_loss_only=prediction_loss_only
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    now = datetime.datetime.now()
    trainer.train()

    print(f"⏰ Training time: {datetime.datetime.now() - now}.")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    results = {}

    eval_output = trainer.evaluate()

    perplexity = math.exp(eval_output["eval_loss"])
    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(output_dir, "eval_results_lm.txt")
    with open(output_eval_file, "w") as writer:
        print("***** Eval results *****")
        for key in sorted(result.keys()):
            print("  %s = %s", key, str(result[key]))

    results.update(result)

    return results


### TRAIN TOKENIZER

VOCAB_SIZE = 52000
MIN_FREQ = 2
PREFIX = 'coberta-mini'


train_tokenizer(
    data_dir='./data/codataset',
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQ,
    output_dir='./models',
    prefix=PREFIX
)

### TRAIN LANGUAGE MODEL

NUM_ATTENTION_HEADS = 4
NUM_HIDDEN_LAYERS = 4
HIDDEN_SIZE = 256
MAX_POSITION_EMBEDDINGS = 512
MAX_LENGTH = 256

train_language_model(
    data_dir='./data/codataset',
    pretrained_path='./models/coberta-mini',
    vocab_size=VOCAB_SIZE,
    max_position_embeddings=MAX_POSITION_EMBEDDINGS,
    num_attention_heads=NUM_ATTENTION_HEADS,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
    type_vocab_size=1,
    max_len=MAX_LENGTH,
    output_dir='./models/coberta-mini',
    num_train_epochs=40,
    per_gpu_train_batch_size=36,
    save_steps=10_000,
    save_total_limit=2
)
