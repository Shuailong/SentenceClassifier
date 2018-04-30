#!/usr/bin/env python
# encoding: utf-8
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Model code.
'''


import logging
import copy

import torch
import torch.optim as optim
import torch.nn.functional as F

from .config import override_model_args
from .classifier import RNNClassifier, CNNClassifier


"""Sent Classifier model"""

logger = logging.getLogger(__name__)


class SentClassifier(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, word_dict, state_dict=None):
        # Book-keeping.
        self.args = args
        self.word_dict = word_dict
        self.args.vocab_size = len(word_dict)
        self.updates = 0
        self.device = None
        self.parallel = False

        # Building network.
        if args.model_type == 'rnn':
            self.network = RNNClassifier(args)
        elif args.model_type == 'cnn':
            self.network = CNNClassifier(args)
        else:
            raise RuntimeError(f'Unsupported model: {args.model_type}')

        if state_dict:
            # Load buffer separately
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')
                self.network.load_state_dict(state_dict)
                self.network.register_buffer(
                    'fixed_embedding', fixed_embedding)
            else:
                self.network.load_state_dict(state_dict)

    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in self.word_dict}
        logger.info(f'Loading pre-trained embeddings for {len(words)} words from {embedding_file}')
        embedding = self.network.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file, encoding='utf-8') as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning(f'WARN: Duplicate embedding found for {w.encode("utf-8")}')
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def init_optimizer(self, state_dict=None):
        """Initialize an optimizer for the free parameters of the network.

        Args:
            state_dict: network parameters
        """
        if self.args.fix_embeddings:
            for p in self.network.embedding.parameters():
                p.requires_grad = False
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay,
                                          lr=self.learning_rate)
        elif self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(parameters,
                                        weight_decay=self.args.weight_decay,
                                        lr=self.args.learning_rate)
        elif self.args.optimizer == 'adadelta':
            self.optimizer = optim.Adadelta(parameters,
                                            weight_decay=self.args.weight_decay,
                                            lr=self.args.learning_rate)
        else:
            raise RuntimeError(f'Unsupported optimizer: {self.args.optimizer}')

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        # Transfer to GPU
        inputs = [e.to(device=self.device) for e in ex[:-1]]
        label = ex[-1].to(device=self.device)
        # Run forward
        score = self.network(*inputs)

        # Compute loss and accuracies
        loss = F.cross_entropy(score, label)

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.network.linear.parameters(),
                                       self.args.grad_clipping)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        return loss.item(), ex[0].size(0)

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex):
        """
        If async_pool is given, these will be AsyncResult handles.
        """
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        inputs = [e.to(self.device) for e in ex]

        with torch.no_grad():
            # Run forward
            score = self.network(*inputs)

        # Decode predictions
        return score.cpu().max(1)[1]

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        state_dict = copy.copy(self.network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'word_dict': self.word_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args=None):
        logger.info(f'Loading model {filename}')
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
        return SentClassifier(args, word_dict, state_dict)

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def to(self, device):
        self.device = device
        self.network = self.network.to(device)

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
