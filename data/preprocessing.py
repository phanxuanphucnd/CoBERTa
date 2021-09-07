# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import re
import ast
import random

from tqdm import tqdm
from unicodedata import normalize as nl

def cleanning_text(text):
    # TODO: Unicode text
    text = nl('NFKC',text)

    # TODO: Remove emotion icon
    emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese char
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642" 
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  # dingbats
                                u"\u3030"
                                u"\u4e00-\u9fff"          # chinese,japan,korean word
                                "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r" ", text) 
    
    # TODO: Remove url
    text=re.sub(r'(http|https|www)?\s*\S+\.\b(vn|com)?\S+', ' ', text)

    # TODO: Remove all html tag and some tag with format word_word ,[],<>
    text = re.sub(r'\[.*?\]|<.*?>|[\w]+_[\w]+', ' ', text)

    # # TODO: Replace numbers patterns
    # text = ' ' + text + ' '
    # text = re.sub(r"(\d*[\s\.\,]?\d+\s?đô la\s+)", " _number_usd ", text)
    # text = re.sub(r"(\d*[\s\.\,]?\d+\s?usd\s+)", " _number_usd ", text)
    # text = re.sub(r"(\d*[\s\.\,]?\d+\s?dollar\s+)", " _number_usd ", text)
    # text = re.sub(r"(\d*[\s\.\,]?\d+\s?$\s+)", " _number_usd ", text)
    # text = re.sub(r"(\d*[\s\.\,]?\d+\s?yen\s+)", " _number_yen ", text)
    # text = re.sub(r"(\d*[\s\.\,]?\d+\s?yên\s+)", " _number_yen ", text)
    # text = re.sub(r"(\d*[\s\.\,]?\d+\s?k\s+)", " _number_vnd ", text)
    # text = re.sub(r"(\d*[\s\.\,]?\d+\s?vnd\s+)", " _number_vnd ", text)
    # text = re.sub(r"(\d*[\s\.\,]?\d+\s?vnđ\s+)", " _number_vnd ", text)
    # text = re.sub(r"(\+?\d{9,11}\s+)", " _number_phone ", text)
    # text = re.sub(r"(\d+\s+)", " _number_ ", text)

    # TODO: Remove puctuation from begining and end string except !.?
    text = re.sub(r'^[^a-zA-Z__ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ0-9]+|[^a-zA-Z__ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ0-9.!?]+$', '', text)

    # TODO: Split word and punctuation
    # text = ' '.join(re.findall("\d+[\+:\-/][^), ]+|\w+\-\S+|\w+|[,.?!]", text))
    text = re.sub(r"\!{1,}", " ! ", text)
    text = re.sub(r"\?{1,}", " ? ", text)
    text = re.sub(r"\.{1,}", " . ", text)
    text = re.sub(r"\,{1,}", " , ", text)
    text = re.sub(r"\({1,}", " ( ", text)
    text = re.sub(r"\){1,}", " ) ", text)
    text = re.sub(r"\-{1,}", " - ", text)
    text = re.sub(r"\~{1,}", " ~ ", text)
    
    # TODO: Remove sequence punctuation continous
    # text = re.sub(r'(?<=[^a-zA-Z__ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ0-9])(?<! )[^a-zA-Z__ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ0-9]+',' ',text)

    # TODO: Remove many whitespace
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip().lower()

def processing_data(data_path, train_path, eval_path, pct=0.01, mode='normal'):
    with open(data_path, 'r+', encoding='utf-8') as rf:
        lines = rf.readlines()

    print(f"The numbers of lines text: {len(lines)}")

    outtexts = []
    for i in tqdm(range(len(lines))):
        if mode == 'fb':
            res = ast.literal_eval(lines[i])
            text = res.get('content', '')
        else:
            text = lines[i]

        if len(text.split(' ')) >= 3:
            text = cleanning_text(text)
            outtexts.append(text)

    NUMBER_SAMPLES = len(outtexts)
    K = int(pct * NUMBER_SAMPLES)
    
    evaltexts = random.sample(outtexts, k=K)
    traintexts = list(set(outtexts) - set(evaltexts))
    print(f"Length train: {len(traintexts)} | eval: {len(evaltexts)}")
    with open(eval_path, 'w', encoding='utf-8') as wf:
        for text in tqdm(evaltexts):
            wf.writelines(text + '\n')

    with open(train_path, 'w', encoding='utf-8') as wf:
        for text in tqdm(traintexts):
            wf.writelines(text + '\n')

data_path = 'wiki2021.txt'
train_path = 'co-dataset/wiki2021-train.txt'
eval_path = 'co-dataset/wiki2021-eval.txt'

# processing_data(data_path, train_path, eval_path, pct=0.001, mode='normal')