# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="models/coberta-tiny/checkpoint-330000",
    tokenizer="models/coberta-tiny/checkpoint-330000"
)

result = fill_mask("giá đt này là <mask>")

from pprint import pprint
pprint(result)

"""
>>> result

  'sequence': 'giá đt này là bn',
  'token': 2044,
  'token_str': ' bn'},
 {'score': 0.08451628684997559,
  'sequence': 'giá đt này là bnhieu',
  'token': 8617,
  'token_str': ' bnhieu'},
 {'score': 0.05901617556810379,
  'sequence': 'giá đt này là sao',
  'token': 644,
  'token_str': ' sao'},
 {'score': 0.03501467779278755,
  'sequence': 'giá đt này là gì',
  'token': 527,
  'token_str': ' gì'},
 {'score': 0.03272506967186928,
  'sequence': 'giá đt này là bnhiu',
  'token': 6631,
  'token_str': ' bnhiu'}]
"""
