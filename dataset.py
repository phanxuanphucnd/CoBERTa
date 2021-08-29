# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import torch

from pathlib import Path
from torch.utils.data import Dataset
from tokenizers.processors import BertProcessing
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

class CoLMDataset(Dataset):
    def __init__(
        self, 
        root: str='./data', 
        mode: str='train', 
        tokenizer: PreTrainedTokenizerBase=None,
        vocab_file: str='./models/CoBERTa-mini/vocab.json',
        merges_file: str='./models/CoBERTa-mini/merges.txt',
        max_length: int=256,
        block_size: int=128
    ) -> None:
        super(CoLMDataset, self).__init__()
        if not tokenizer:
            tokenizer = ByteLevelBPETokenizer(
                vocab_file,
                merges_file
            )
            tokenizer._tokenizer.post_processor = BertProcessing(
                ('</s>', tokenizer.token_to_id('</s>')),
                ('<s>', tokenizer.token_to_id('<s')),
            )
            tokenizer.enable_truncation(max_length=max_length)

        self.examples = []

        src_files = Path(root).glob(f'*-{mode}.txt')
        for src_file in src_files:
            print("🔥", src_file)
            lines = src_file.read_text(encoding='utf-8').splitlines()
            self.examples += [x.ids for x in tokenizer.encode_batch(lines)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])
