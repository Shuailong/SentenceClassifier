# Sentence Classifier
Sentence level classification implementation based on PyTorch.

### Features
- CNN model from Yoonkim
- RNN attention/avg pooling model with/without merged RNN layers
- GloVE / word2vec embedding
- if dev data missing, random select 10% of training set as dev
- if test data missing, use 10 fold cross validation

### Performance

| Dataset | CNN-rand | CNN-w2v   | CNN-glove | RNN-rand | RNN-w2v | RNN-glove |
|:-----------|:------------:|:-------------:|:-------------:|:------------:|:-----------:|:-------------:|
| __MR__   | 75.90 (76.1)|  __81.18__(81.5)|    78.84    |  73.69     |   79.95   |     79.33   |
| __TREC__   | 91.20 (91.2)|  93.80(93.6)|    __94.60__    |  88.80     |   90.80   |     89.40   |
| __SST-5__   | 44.75 (45.0)|  46.47(48.0)|    44.43    |  44.52     |   __51.54__   |     47.83   |
| __SST-2__   | 84.29 (82.7)|  86.66(87.2)|    84.95    |  82.37     |   __87.59__   |     86.88   |

*\*Numbers in brackets are from Yoonkim's paper*
*\*Hyperparameter the same as Yoonkim. Default random seed, run once.*
*\*MR: 10 fold cross validation. TREC: random dev set.*
*\*RNN models are not tuned and use the same hyperparams as CNN models*

### Usage

Preprocess dataset
```
python script/preprocess/prepare_sst.py --help
```
Train on the dataset
```
python script/classify/train.py --help
```

### Demo

```bash
python script/classify/prepare_sst.py --uncased --phrase --n-class 5
```

```bash
python script/classify/train.py --model-type cnn --n-class 5 --train-file sst/sst5_train.json --dev-file sst/sst5_dev.json --test-file sst/sst5_test.json --display-iter 500 --embedding-file word2vec --num-epoch 5
```

```bash
04/30/2018 01:23:45 PM: [ COMMAND: script/classify/train.py --model-type rnn --n-class 5 --train-file sst/sst5_train.json --dev-file sst/sst5_dev.json --test-file sst/sst5_test.json --display-iter 500 --embedding-file word2vec --num-epoch 5 ]
04/30/2018 01:23:45 PM: [ ---------------------------------------------------------------------------------------------------- ]
04/30/2018 01:23:45 PM: [ Load data files ]
04/30/2018 01:23:45 PM: [ Num train examples = 151765 ]
04/30/2018 01:23:45 PM: [ Num dev examples = 1100 ]
04/30/2018 01:23:45 PM: [ Num test examples = 2210 ]
04/30/2018 01:23:45 PM: [ Total 155075 examples. ]
04/30/2018 01:23:45 PM: [ ---------------------------------------------------------------------------------------------------- ]
04/30/2018 01:23:45 PM: [ CONFIG:
{
    "batch_size": 50,
    "concat_rnn_layers": false,
    "cuda": true,
...
} ]
...
04/30/2018 01:23:45 PM: [ Build dictionary ]
04/30/2018 01:23:46 PM: [ Num words = 17838 ]
04/30/2018 01:23:46 PM: [ Loading pre-trained embeddings for 17836 words from /home/lsl/BlamePipeline/data/embeddings/w2v.googlenews.300d.txt ]
04/30/2018 01:24:25 PM: [ Loaded 16262 embeddings (91.18%) ]
04/30/2018 01:24:27 PM: [ ---------------------------------------------------------------------------------------------------- ]
04/30/2018 01:24:27 PM: [ train: Epoch = 0 | iter = 0/3036 | loss = 1.59 | elapsed time = 0.06 (s) ]
04/30/2018 01:24:30 PM: [ train: Epoch = 0 | iter = 500/3036 | loss = 1.08 | elapsed time = 3.37 (s) ]
...
04/30/2018 01:24:47 PM: [ train: Epoch 0 done. Time for epoch = 20.14 (s) ]
04/30/2018 01:24:48 PM: [ train valid: Epoch = 0 (best:0) | examples = 10000 | valid time = 0.69 (s). ]
04/30/2018 01:24:48 PM: [ acc: 61.49% ]
04/30/2018 01:24:48 PM: [ dev valid: Epoch = 0 (best:0) | examples = 1100 | valid time = 0.11 (s). ]
04/30/2018 01:24:48 PM: [ acc: 46.09% ]
04/30/2018 01:24:48 PM: [ Best valid: acc = 46.09% (epoch 0, 3036 updates) ]
...
04/30/2018 01:26:13 PM: [ ---------------------------------------------------------------------------------------------------- ]
04/30/2018 01:26:13 PM: [ Load best model... ]
04/30/2018 01:26:13 PM: [ Loading model /home/lsl/BlamePipeline/data/models/20180430-5302c404.mdl ]
04/30/2018 01:26:13 PM: [ test valid: Epoch = 3 (best:3) | examples = 2210 | valid time = 0.21 (s). ]
04/30/2018 01:26:13 PM: [ acc: 51.54% ]
04/30/2018 01:26:13 PM: [ ---------------------------------------------------------------------------------------------------- ]
04/30/2018 01:26:13 PM: [ Test acc: 51.54% ]
```

### Requirements
- python>=3.6
- pytorch>=0.4
- termcolor
- tqdm
- pytreebank (generate data for SST)
- gensim (convert word2vec bin to glove format)

### Installation

```
python setup.py develop
```

### Data
- SST: [Stanford Sentiment](https://nlp.stanford.edu/sentiment/index.html)
- TREC & MR: [sent-conv-torch](https://github.com/harvardnlp/sent-conv-torch)
- glove: [Stanford GloVe Project](https://nlp.stanford.edu/projects/glove/)
- word2vec: [Google Code Archive](https://code.google.com/archive/p/word2vec/)

##### Process word2vec
```python
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
model.wv.save_word2vec_format('w2v.googlenews.300d.txt')
```
##### Remove first line
```bash
tail -n +2 w2v.googlenews.300d.txt > w2v.googlenews.300d.txt.new && mv -f w2v.googlenews.300d.txt.new w2v.googlenews.300d.txt
```

### References
- [yoonkim/CNN_sentence](https://github.com/yoonkim/CNN_sentence) (preprocessing, model hyperparameters)
- [Shawn1993/cnn-text-classification-pytorch](https://github.com/Shawn1993/cnn-text-classification-pytorch) (CNN pytorch implementation)
- [facebookresearch/DrQA](https://github.com/facebookresearch/DrQA) (code archtecture, RNN layer implementation)
- [harvard-nlp/sent-conv-torch](https://github.com/harvardnlp/sent-conv-torch) (processed data)

