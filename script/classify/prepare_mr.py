# coding: utf-8

'''
Generate training samples for end2end blame extraction.
'''

import os
import re
import json
import logging
import argparse

from blamepipeline import DATA_DIR

DATASET = os.path.join(DATA_DIR, 'datasets')
logger = logging.getLogger()


def stat(sents):

    lens = [len(sent) for sent in sents]

    min_len = min(lens)
    avg_len = int(sum(lens) / len(lens))
    max_len = max(lens)

    logger.info(f'min/avg/max sent length: {min_len}/{avg_len:.0f}/{max_len}')


def clean_mr(sent):
    sent = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sent)
    sent = re.sub(r"\'s", " \'s", sent)
    sent = re.sub(r"\'ve", " \'ve", sent)
    sent = re.sub(r"n\'t", " n\'t", sent)
    sent = re.sub(r"\'re", " \'re", sent)
    sent = re.sub(r"\'d", " \'d", sent)
    sent = re.sub(r"\'ll", " \'ll", sent)
    sent = re.sub(r",", " , ", sent)
    sent = re.sub(r"!", " ! ", sent)
    sent = re.sub(r"\(", " \( ", sent)
    sent = re.sub(r"\)", " \) ", sent)
    sent = re.sub(r"\?", " \? ", sent)
    sent = re.sub(r"\s{2,}", " ", sent)
    return sent.strip().lower()


def main(args):
    logger.info('loading rt-polarity Movie Review data')

    data = []
    total_samples = 0

    dataset_file = os.path.join(DATASET, 'mr', f'rt-polarity.all')
    with open(dataset_file, encoding='latin-1') as f:
        lines = 0
        for line in f:
            lines += 1
            tokens = line.split()
            label = int(tokens[0])
            sentence = ' '.join(tokens[1:])
            sentence = clean_mr(sentence)
            data.append((sentence, label))

    logger.info('-' * 100)
    logger.info(f'{lines} sentences generates {len(data)} examples.')
    total_samples += len(data)
    logger.info('calculate sentence statistics')
    for i, (sent, label) in enumerate(data):
        data[i] = (sent.split(), label)

    stat([sent for sent, label in data])
    sent_file = os.path.join(DATASET, 'mr', f'rt-polarity.json')
    logger.info(f'write samples to {sent_file}')

    with open(sent_file, 'w') as f:
        for sent, label in data:
            line = json.dumps({
                'label': label,
                'sent': sent
            })
            f.write(line + '\n')


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    parser = argparse.ArgumentParser('Generate dataset from https://github.com/harvardnlp/sent-conv-torch')
    parser.set_defaults()
    args = parser.parse_args()
    main(args)
