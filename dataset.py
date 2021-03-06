# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import os
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
        max_length: int=256,
        block_size: int=128
    ) -> None:
        super(CoLMDataset, self).__init__()
        self.examples = []
        # all_lines = []

        # src_files = Path(root).glob(f'*-{mode}.txt')
        print(f"MODE: {mode.upper()}")
        # for i, src_file in enumerate(src_files):
        for i, file in enumerate(os.listdir(root)):
            if mode in file and file.endswith('.txt'):
                src_file = f"{root}/{file}"
                print(f"📄 {i}: ", src_file)
                # lines = src_file.read_text(encoding='utf-8').splitlines()
                with open(src_file, 'r+', encoding='utf-8') as f:
                    lines = f.read().splitlines()
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
                    # all_lines.extend(lines)
                    batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
                    self.examples.extend([{"input_ids": torch.tensor(e, dtype=torch.long)} for e in batch_encoding["input_ids"]])

                    del lines
                    del batch_encoding
            
        # if tokenizer:
        #     batch_encoding = tokenizer(all_lines, add_special_tokens=True, truncation=True, max_length=block_size)
        #     self.examples = batch_encoding["input_ids"]

        # self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
