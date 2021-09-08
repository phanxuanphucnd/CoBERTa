# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import os
import math
import argparse
import datetime

from pathlib import Path
from dataset import CoLMDataset
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizerFast
from transformers import LineByLineTextDataset
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from tokenizers.implementations import ByteLevelBPETokenizer

def train_tokenizer(
    data_dir: str='./data', 
    vocab_size: int=52000, 
    min_frequency: int=2, 
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
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>"
        ]
    )

    # TODO: Save files
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    
    tokenizer.save_model(prefix)

def train_language_model(
    data_dir: str='./data',
    pretrained_path: str='./models/coberta-mini',
    vocab_size: int=52000,
    max_position_embeddings: int=512,
    num_attention_heads: int=12,
    num_hidden_layers: int=6,
    hidden_size: int=256,
    type_vocab_size: int=2,
    max_len: int=256,
    block_size: int=128,
    output_dir: str='./models/coberta-mini',
    learning_rate: float=5e-4,
    num_train_epochs: int=40,
    per_device_train_batch_size: int=128,
    save_steps: int=10_000,
    save_total_limit: int=2,
    prediction_loss_only: bool=True,
    **kwargs
):
    config = RobertaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        type_vocab_size=type_vocab_size,
    )

    tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_path, max_len=max_len)

    model = RobertaForMaskedLM(config=config)
    print(f"\n🔥 The numbers of parameters: {model.num_parameters()}\n")

    train_dataset = CoLMDataset(
        root=data_dir, 
        mode='train',
        tokenizer=tokenizer,
        max_length=max_len,
    )
    eval_dataset = CoLMDataset(
        root=data_dir,
        mode='eval',
        tokenizer=tokenizer,
        max_length=max_len,
    )
    print(f"\n---------- DATASET INFO ----------")
    print(f"The length of Train Dataset: {len(train_dataset)}")
    print(f"The length of Eval Dataset: {len(eval_dataset)}")
    print(f"----------------------------------")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        prediction_loss_only=prediction_loss_only
    )
    print(f"\n---------- TRAINING PRETRAIN LANGUAGE MODEL ----------")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    now = datetime.datetime.now()
    trainer.train()

    print(f"\n⏰ Training time: {datetime.datetime.now() - now}.")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    results = {}

    results = trainer.evaluate()

    perplexity = math.exp(results["eval_loss"])
    results["perplexity"] = perplexity

    from pprint import pprint
    pprint(results)

    output_eval_file = os.path.join(output_dir, "eval_results_lm.txt")
    with open(output_eval_file, "w") as writer:
        print("***** Eval results *****")
        for key, value in sorted(results.items()):
            print(f" {key} = {value}")
            writer.write(f"{key} = {value}\n")

    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument("--train_tokenizer", action='store_true', 
                        help="Setup mode `train tokenizer`.")
    parser.add_argument("--train_lm", action='store_true', 
                        help="Setup mode `train language model`.")
    parser.add_argument("--dataset_path", type=str, default='data/co-dataset',
                        help="The path to the dataset to use.",)
    parser.add_argument("--num_train_epochs", type=int, default=1, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=123, 
                        help="A seed for reproducible training.")

    args = parser.parse_args()

    VOCAB_SIZE = 52000
    MIN_FREQ = 2
    PREFIX = 'coberta-mini'

    NUM_ATTENTION_HEADS = 4
    NUM_HIDDEN_LAYERS = 4
    HIDDEN_SIZE = 256
    MAX_POSITION_EMBEDDINGS = 514
    MAX_LENGTH = 128
    LEARNING_RATE = 5e-4

    ### TRAIN TOKENIZER
    if args.train_tokenizer:
        print("\nTRAINING TOKENIZER...")
        train_tokenizer(
            data_dir=args.dataset_path,
            vocab_size=VOCAB_SIZE,
            min_frequency=MIN_FREQ,
            prefix=PREFIX
        )
    
    ### TRAIN LANGUAGE MODEL
    if args.train_lm:
        print("\nTRAINING PRE_TRAINING LANGUAGE MODEL...")
        train_language_model(
            data_dir=args.dataset_path,
            pretrained_path='coberta-mini',
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            num_attention_heads=NUM_ATTENTION_HEADS,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            type_vocab_size=2,
            hidden_size=HIDDEN_SIZE,
            learning_rate=LEARNING_RATE, 
            max_len=MAX_LENGTH,
            output_dir='./models/coberta-mini',
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=64,
            save_steps=10_000,
            save_total_limit=2
        )
