#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Functions for putting examples into torch format."""

import torch


def vectorize(ex, model):
    """Torchify a single example."""
    word_dict = model.word_dict
    # Index words
    word_idx = [word_dict[w] for w in ex['sent']]

    if len(word_idx) == 0:
        print(ex)

    if model.args.model_type == 'cnn':
        pad = max(model.args.kernel_sizes) - 1
        word_idx = [0] * pad + word_idx + [0] * pad
    sent = torch.tensor(word_idx, dtype=torch.long)

    # Maybe return without target
    if 'label' not in ex:
        return sent
    else:
        return sent, ex['label']


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    if isinstance(batch[0], tuple):
        pred_mode = False
        sents = [ex[0] for ex in batch]
        labels = [ex[-1] for ex in batch]
        y = torch.tensor(labels, dtype=torch.long)
    else:
        pred_mode = True
        sents = batch

    max_length = max([s.size(0) for s in sents])
    x = torch.zeros(len(sents), max_length, dtype=torch.long)
    x_mask = torch.ones(len(sents), max_length, dtype=torch.uint8)

    for i, s in enumerate(sents):
        x[i, :s.size(0)].copy_(s)
        x_mask[i, :s.size(0)].fill_(0)

    # Maybe return without targets
    if pred_mode:
        return x, x_mask
    else:
        return x, x_mask, y
