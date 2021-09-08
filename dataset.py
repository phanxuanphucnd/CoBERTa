# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import torch

from typing import Dict
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
        vocab_file: str='coberta-mini/vocab.json',
        merges_file: str='coberta-mini/merges.txt',
        max_length: int=128,
        block_size: int=128
    ) -> None:
        super(CoLMDataset, self).__init__()
        self.examples = []
        all_lines = []

        src_files = Path(root).glob(f'*-{mode}.txt')
        print(f"MODE: {mode.upper()}")
        for src_file in src_files:
            print("📄", src_file)
            lines = src_file.read_text(encoding='utf-8').splitlines()
            lines = list(filter(None, lines))
            if not tokenizer:
                tokenizer = ByteLevelBPETokenizer(
                    vocab_file,
                    merges_file
                )
                tokenizer._tokenizer.post_processor = BertProcessing(
                    ('</s>', tokenizer.token_to_id('</s>')),
                    ('<s>', tokenizer.token_to_id('<s>')),
                )
                tokenizer.enable_truncation(max_length=max_length)
                
                self.examples += [x.ids for x in tokenizer.encode_batch(lines)]
            else:
                all_lines.extend(lines)
        
        if tokenizer:
            batch_encoding = tokenizer(all_lines, add_special_tokens=True, truncation=True, max_length=block_size)
            self.examples = batch_encoding["input_ids"]
    
        # TODO: Free memory of somes variable
        del batch_encoding
        del all_lines

        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
