### Table of contents

1. [Introduction](#introduction)
2. [How to use ``CosoBERTa``](#how_to_use_soberta)
    - [Installation](#installation)
    - [Pre-trained models](#models)
    - [Example usage](#usage)


# <a name='introduction'></a> CoBERTa

CosoBERTa is a pre-trained models are the pretrained language models for Comment/ Social Vietnamese dataset:

 - Two CosoBERTa versions of `mini` and `small` are first public large-scale monolingual language models pre-trained for Comment/ Social Vietnamese. CosoBERTa pre-training approach is based on [RoBERTa](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md).
 - CosoBERTa is the basic for improve the performance of downstream Vietnames tasks in Social Networks.


# <a name='how_to_use_aubbert'></a> How to use AuBBERT

## Installation <a name='installation'></a>

 - Python 3.6+, and Pytorch 1.4.0+ 

## Pre-trained models <a name='models'></a>

Model | #params | Arch.	 | Pre-training data
---|---|---|---
`cosoberta-mini` | 50M | mini | 5GB of texts
`cosoberta-small` | 70M | small | 5GB of texts

## License

    MIT License

    Copyright (c) 2021 Phuc Phan

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

## Author

- Name: Phan Xuan Phuc
- Email: phanxuanphucnd@gmail.com