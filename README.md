### Table of contents

1. [Introduction](#introduction)
2. [How to use ``CoBERTa``](#how_to_use_coberta)
    - [Installation](#installation)
    - [Pre-trained models](#models)
    - [Example usage](#usage)


# <a name='introduction'></a> CoBERTa

How to train a tranformers-based language model from a custom dataset?

`CoBERTa` is a pre-trained models are the pre-trained language models for Comment-in-social/ Conversation Vietnamese datasets:

 - Two `CoBERTa` versions of `mini` and `small` are first public large-scale monolingual language models pre-trained for Comment-in-Social/ Conversation Vietnamese. `CoBERTa` pre-training approach is based on [RoBERTa](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md).
 - `CoBERTa` is the basic for improve the performance of downstream Vietnames tasks in Social Networks.


# <a name='how_to_use_coberta'></a> How to use CoBERTa

## Installation <a name='installation'></a>

 - Python 3.6+, and Pytorch 1.4.0+ 

## Pre-trained models <a name='models'></a>

Model | #params | Arch.	 | Pre-training data
---|---|---|---
`coberta-mini` | 21M | mini | 4.3GB of texts
`coberta-small` | 70M | small | 4.3GB of texts

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
