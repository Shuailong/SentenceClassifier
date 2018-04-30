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

from blamepipeline import DATA_DIR
import pytreebank

DATASET = os.path.join(DATA_DIR, 'datasets')
logger = logging.getLogger()


def stat(sents):

    lens = [len(sent) for sent in sents]

    min_len = min(lens)
    avg_len = int(sum(lens) / len(lens))
    max_len = max(lens)

    logger.info(f'min/avg/max sent length: {min_len}/{avg_len:.0f}/{max_len}')


def clean_sst(sent, uncased=True):
    sent = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sent)
    sent = re.sub(r"\s{2,}", " ", sent)
    sent = sent.strip()
    if uncased:
        sent = sent.lower()
    return sent


def main(args):
    logger.info('loading SST data')
    dataset = pytreebank.load_sst(os.path.join(DATASET, 'sst'))

    split = ['train', 'dev', 'test']
    data = defaultdict(set)
    total_samples = 0
    for tag in split:
        for example in dataset[tag]:
            for label, sentence in example.to_labeled_lines():
                sentence = clean_sst(sentence, uncased=args.uncased)
                if args.n_class == 5:
                    if (sentence, label) not in data[tag] and sentence:
                        data[tag].add((sentence, label))
                else:
                    if label > 2:
                        if (sentence, 1) not in data[tag] and sentence:
                            data[tag].add((sentence, 1))
                    elif label < 2:
                        if (sentence, 0) not in data[tag] and sentence:
                            data[tag].add((sentence, 0))
                if args.phrase and tag == 'train':
                    continue
                else:
                    break
        logger.info('-' * 100)
        logger.info(f'{tag}: {len(dataset[tag])} sentences generates {len(data[tag])} examples.')
        total_samples += len(data[tag])
        logger.info('calculate sentence statistics')
        data[tag] = list(data[tag])
        for i, (sent, label) in enumerate(data[tag]):
            data[tag][i] = (sent.split(), label)

        stat([sent for sent, label in data[tag]])
        sent_file = os.path.join(DATASET, 'sst', f'sst{args.n_class}_{tag}.json')
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

    parser = argparse.ArgumentParser('Generate SST dataset from CoNLL format')
    parser.add_argument('--n-class', type=int, default=2, choices=[2, 5],
                        help='SST 2 or 5')
    parser.add_argument('--uncased', action='store_true',
                        help='whether to lowercase the sentence')
    parser.add_argument('--phrase', action='store_true',
                        help='add phrase level training instances')
    parser.set_defaults()
    args = parser.parse_args()
    main(args)
