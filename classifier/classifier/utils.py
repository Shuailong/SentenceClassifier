#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""DrQA reader utilities."""

import json
import time
import logging
import random
from collections import Counter

import torch

from .data import Dictionary
from .data import SentenceDataset
from . import vector

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Train/dev split
# ------------------------------------------------------------------------------

def split_loader(train_exs, test_exs, args, model, dev_exs=None):
    train_dataset = SentenceDataset(train_exs, model)
    train_size = len(train_dataset)
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda)

    test_dataset = SentenceDataset(test_exs, model)
    test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        sampler=test_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda)

    if dev_exs:
        dev_dataset = SentenceDataset(dev_exs, model)
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
        dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=args.test_batch_size,
            sampler=dev_sampler,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda)
    else:
        dev_size = int(train_size * 0.1)
        train_dev_idxs = list(range(train_size))
        random.shuffle(train_dev_idxs)
        dev_idxs = train_dev_idxs[-dev_size:]
        train_idxs = train_dev_idxs[:train_size - dev_size]
        dev_sampler = torch.utils.data.sampler.SubsetRandomSampler(dev_idxs)
        dev_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.test_batch_size,
            sampler=dev_sampler,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda)
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idxs)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda)

    return train_loader, dev_loader, test_loader


def split_loader_cv(train_exs, args, model, dev_idxs):
    train_dataset = SentenceDataset(train_exs, model)
    train_idxs = set(range(len(train_dataset))) - set(dev_idxs)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(list(train_idxs))
    dev_sampler = torch.utils.data.sampler.SubsetRandomSampler(dev_idxs)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda)
    dev_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda)
    return train_loader, dev_loader

# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------


def load_data(filename):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    """
    # Load JSON lines
    with open(filename) as f:
        examples = [json.loads(line) for line in f]

    return examples


# ------------------------------------------------------------------------------
# Dictionary building
# ------------------------------------------------------------------------------


def load_words(args, examples):
    """Iterate and index all the words in examples (documents + questions)."""

    words = Counter()
    for ex in examples:
        for w in ex['sent']:
            w = Dictionary.normalize(w)
            words[w] += 1
    words = (w for w, _ in words.most_common())
    return words


def build_word_dict(args, examples):
    """Return a dictionary from sentence words in
    provided examples.
    """
    word_dict = Dictionary()
    for w in load_words(args, examples):
        word_dict.add(w)
    return word_dict


# ------------------------------------------------------------------------------
# Utility classes
# ------------------------------------------------------------------------------


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total
