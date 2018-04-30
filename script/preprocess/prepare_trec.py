# coding: utf-8

'''
Generate training samples for end2end blame extraction.
'''

import os
import re
import json
import logging
import argparse
from collections import defaultdict

from classifier import DATA_DIR

DATASET = os.path.join(DATA_DIR, 'datasets')
logger = logging.getLogger()


def stat(sents):

    lens = [len(sent) for sent in sents]

    min_len = min(lens)
    avg_len = int(sum(lens) / len(lens))
    max_len = max(lens)

    logger.info(f'min/avg/max sent length: {min_len}/{avg_len:.0f}/{max_len}')


def clean_trec(sent):
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
    return sent.strip()


def main(args):
    logger.info('loading Trec data')

    split = ['train', 'test']

    data = defaultdict(list)
    total_samples = 0
    for tag in split:
        dataset_file = os.path.join(DATASET, 'trec', f'TREC.{tag}.all')
        with open(dataset_file, encoding='latin-1') as f:
            lines = 0
            for line in f:
                lines += 1
                tokens = line.split()
                label = int(tokens[0])
                sentence = ' '.join(tokens[1:])
                sentence = clean_trec(sentence)
                data[tag].append((sentence, label))

        logger.info('-' * 100)
        logger.info(f'{tag}: {lines} sentences generates {len(data[tag])} examples.')
        total_samples += len(data[tag])
        logger.info('calculate sentence statistics')
        for i, (sent, label) in enumerate(data[tag]):
            data[tag][i] = (sent.split(), label)

        stat([sent for sent, label in data[tag]])
        sent_file = os.path.join(DATASET, 'trec', f'trec_{tag}.json')
        logger.info(f'write samples to {sent_file}')

        with open(sent_file, 'w') as f:
            for sent, label in data[tag]:
                line = json.dumps({
                    'label': label,
                    'sent': sent
                })
                f.write(line + '\n')
    logger.info('-' * 100)


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
