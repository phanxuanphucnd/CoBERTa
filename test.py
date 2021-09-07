# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="models/coberta-mini",
    tokenizer="models/coberta-mini"
)

result = fill_mask("giá đt này là <mask>")

from pprint import pprint
pprint(result)