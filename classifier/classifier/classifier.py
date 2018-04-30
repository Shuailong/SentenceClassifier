#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Implementation of the Sentence Classifier."""

import torch.nn as nn
from . import layers


# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------

class RNNClassifier(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args):
        super(RNNClassifier, self).__init__()
        # Store config
        self.args = args

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = args.embedding_dim
        # RNN sentence encoder
        self.sent_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # Output sizes of rnn encoders
        sent_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            sent_hidden_size *= args.layers

        # sentence merging
        if args.sent_merge not in ['avg', 'self_attn']:
            raise NotImplementedError('merge_mode = %s' % args.merge_mode)
        if args.sent_merge == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(sent_hidden_size)

        self.linear = nn.Linear(sent_hidden_size, args.n_class)

    def forward(self, x, x_mask):
        """Inputs:
        x = sentence word indices             [batch * len_d]
        x_mask = sentence padding mask        [batch * len_d]
        """
        # Embed both sentence and question
        x_emb = self.embedding(x)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x_emb = nn.functional.dropout(x_emb, p=self.args.dropout_emb,
                                          training=self.training)
        # Form sentence encoding inputs
        drnn_input = x_emb
        # Encode sentence with RNN
        sent_hiddens = self.sent_rnn(drnn_input, x_mask)

        if self.args.sent_merge == 'avg':
            merge_weights = layers.uniform_weights(sent_hiddens, x_mask)
        elif self.args.sent_merge == 'self_attn':
            merge_weights = self.self_attn(sent_hiddens, x_mask)
        sent_repr = layers.weighted_avg(sent_hiddens, merge_weights)

        scores = self.linear(sent_repr)

        return scores


class CNNClassifier(nn.Module):
    def __init__(self, args):
        super(CNNClassifier, self).__init__()
        # Store config
        self.args = args

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)
        self.embedding.weight.data.uniform_(-0.01, 0.01)

        # CNN sentence encoder
        self.sent_cnn = layers.CNN2(args)
        # Output sizes of rnn encoders

        self.linear = nn.Linear(args.hidden_size * len(args.kernel_sizes), args.n_class)

    def forward(self, x, x_mask):
        """Inputs:
        x = sentence word indices             [batch * len_d]
        x_mask = sentence padding mask        [batch * len_d]
        """
        # Embed both sentence and question
        x_emb = self.embedding(x)
        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x_emb = nn.functional.dropout(x_emb, p=self.args.dropout_emb,
                                          training=self.training)
        # Form sentence encoding inputs
        sent_repr = self.sent_cnn(x_emb)

        if self.args.dropout_cnn > 0:
            sent_repr = nn.functional.dropout(sent_repr, p=self.args.dropout_cnn,
                                              training=self.training)

        scores = self.linear(sent_repr)

        return scores
