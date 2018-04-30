# !/usr/bin/env python
# encoding: utf-8
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Sentence classifier"""

import argparse
import json
import os
import sys
import subprocess
import logging
from collections import defaultdict

from termcolor import colored
import random
import numpy as np
import torch

from blamepipeline import DATA_DIR as DATA_ROOT
from blamepipeline.claimclass import SentClassifier
from blamepipeline.claimclass import utils, config


logger = logging.getLogger()


# ------------------------------------------------------------------------------
# Training arguments.
# ------------------------------------------------------------------------------


# Defaults
DATA_DIR = os.path.join(DATA_ROOT, 'datasets')
MODEL_DIR = os.path.join(DATA_ROOT, 'models')
EMBED_DIR = os.path.join(DATA_ROOT, 'embeddings')


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--no-cuda', type='bool', default=False,
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', type=int, default=0,
                         help='Run on a specific GPU')
    runtime.add_argument('--data-workers', type=int, default=1,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--parallel', type='bool', default=False,
                         help='Use DataParallel on all available GPUs')
    runtime.add_argument('--random-seed', type=int, default=712,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num-epochs', type=int, default=25,
                         help='Train data iterations')
    runtime.add_argument('--batch-size', type=int, default=50,
                         help='Batch size for training')
    runtime.add_argument('--test-batch-size', type=int, default=50,
                         help='Batch size during validation/testing')

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--model-dir', type=str, default=MODEL_DIR,
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model-name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data-dir', type=str, default=DATA_DIR,
                       help='Directory of training/validation data')
    files.add_argument('--train-file', type=str, default=None,
                       help='train file')
    files.add_argument('--dev-file', type=str, default=None,
                       help='dev file')
    files.add_argument('--test-file', type=str, default=None,
                       help='test file')
    files.add_argument('--embed-dir', type=str, default=EMBED_DIR,
                       help='Directory of pre-trained embedding files')
    files.add_argument('--embedding-file', type=str, choices=['word2vec', 'glove'],
                       default=None, help='Space-separated pretrained embeddings file')
    files.add_argument('--valid-size', type=float, default=0,
                       help='validation set ratio')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--display-iter', type=int, default=25,
                         help='Log state after every <display_iter> batches')
    general.add_argument('--metrics', type=str, choices=['precision', 'recall', 'F1', 'acc'],
                         help='metrics to display when training', nargs='+',
                         default=['acc'])
    general.add_argument('--valid-metric', type=str, default='acc',
                         help='The evaluation metric used for model selection')


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    args.train_file = os.path.join(args.data_dir, args.train_file)
    if not os.path.isfile(args.train_file):
        raise IOError('No such file: %s' % args.train_file)

    if args.dev_file:
        args.dev_file = os.path.join(args.data_dir, args.dev_file)
        if not os.path.isfile(args.dev_file):
            raise IOError('No such file: %s' % args.dev_file)

    if args.test_file:
        args.test_file = os.path.join(args.data_dir, args.test_file)
        if not os.path.isfile(args.test_file):
            raise IOError('No such file: %s' % args.test_file)

    if args.embedding_file:
        args.embedding_file = 'w2v.googlenews.300d.txt' if args.embedding_file == 'word2vec' else 'glove.6B.300d.txt'
        args.embedding_file = os.path.join(args.embed_dir, args.embedding_file)
        if not os.path.isfile(args.embedding_file):
            raise IOError('No such file: %s' % args.embedding_file)

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    args.log_file = os.path.join(args.model_dir, args.model_name + '.txt')
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')

    # Embeddings options
    if args.embedding_file:
        with open(args.embedding_file) as f:
            dim = len(f.readline().strip().split(' ')) - 1
        args.embedding_dim = dim
    elif not args.embedding_dim:
        raise RuntimeError('Either embedding_file or embedding_dim '
                           'needs to be specified.')

    # Make sure fix_embeddings and embedding_file are consistent
    if args.fix_embeddings:
        if not args.embedding_file:
            logger.warning('WARN: fix_embeddings set to False '
                           'as embeddings are random.')
            args.fix_embeddings = False

    return args


# ------------------------------------------------------------------------------
# Initalization from scratch.
# ------------------------------------------------------------------------------


def init_from_scratch(args, train_exs, dev_exs, test_exs):
    """New model, new data, new dictionary.
    """

    # Build a dictionary from the data
    logger.info('-' * 100)
    logger.info('Build dictionary')
    word_dict = utils.build_word_dict(args, train_exs + dev_exs + test_exs)
    logger.info('Num words = %d' % len(word_dict))

    # Initialize model
    model = SentClassifier(config.get_model_args(args), word_dict)

    # Load pretrained embeddings for words in dictionary
    if args.embedding_file:
        model.load_embeddings(word_dict.tokens(), args.embedding_file)

    return model


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------

def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    # Initialize meters + timers
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()

    # Run one epoch
    for idx, ex in enumerate(data_loader):
        loss, batch_size = model.update(ex)
        train_loss.update(loss, batch_size)
        # train_loss.update(*model.update(ex))

        if idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], idx, len(data_loader)) +
                        'loss = %.2f | elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()

    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                (global_stats['epoch'], epoch_time.time()))

# ------------------------------------------------------------------------------
# Validation loops. Includes functions that
# use different metrics and implementations.
# ------------------------------------------------------------------------------


def evaluate(pred, true, eps=1e-9):
    true_positive = (pred * true).sum().item()
    precision = true_positive / (pred.sum().item() + eps)
    recall = true_positive / (true.sum().item() + eps)
    F1 = 2 * (precision * recall) / (precision + recall + eps)
    acc = (pred == true).sum().item() / pred.size(0)
    return {'precision': precision, 'recall': recall, 'F1': F1, 'acc': acc}


def validate(args, data_loader, model, global_stats, mode):
    """Run one full validation.
    """
    eval_time = utils.Timer()
    meters = {name: utils.AverageMeter() for name in args.metrics}

    # Make predictions
    examples = 0

    for ex in data_loader:
        batch_size = ex[0].size(0)
        inputs = ex[:-1]
        pred = model.predict(inputs)
        true = ex[-1]
        # We get metrics for independent start/end and joint start/end
        metrics = evaluate(pred, true)
        for name in args.metrics:
            meters[name].update(metrics[name], batch_size)

        # If getting train accuracies, sample max 10k
        examples += batch_size
        if mode == 'train' and examples >= 1e4:
            break

    logger.info(f'{mode} valid: Epoch = {global_stats["epoch"]} (best:{global_stats["best_epoch"]}) | ' +
                f'examples = {examples} | valid time = {eval_time.time():.2f} (s).')
    logger.info(' | '.join([f'{k}: {meters[k].avg*100:.2f}%' for k in meters]))

    return {args.valid_metric: meters[args.valid_metric].avg}


def train_valid_loop(train_loader, dev_loader, args, model, test_loader=None, fold=None):
    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    logger.info('-' * 100)
    stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0, 'best_epoch': 0}
    start_epoch = 0
    for epoch in range(start_epoch, args.num_epochs):
        stats['epoch'] = epoch

        # Train
        train(args, train_loader, model, stats)

        # Validate train
        validate(args, train_loader, model, stats, mode='train')

        # Validate dev
        result = validate(args, dev_loader, model, stats, mode='dev')

        # Save best valid
        if result[args.valid_metric] > stats['best_valid']:
            logger.info(
                colored(f'Best valid: {args.valid_metric} = {result[args.valid_metric]*100:.2f}% ', 'yellow') +
                colored(f'(epoch {stats["epoch"]}, {model.updates} updates)', 'yellow'))
            fold_info = f'.fold_{fold}' if fold else ''
            model.save(args.model_file + fold_info)
            stats['best_valid'] = result[args.valid_metric]
            stats['best_epoch'] = epoch
        logger.info('-' * 100)

    logger.info('Load best model...')
    model = SentClassifier.load(args.model_file + fold_info, args)
    device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")
    model.to(device)
    stats['epoch'] = stats['best_epoch']
    if test_loader:
        test_result = validate(args, test_loader, model, stats, mode='test')
    else:
        test_result = validate(args, dev_loader, model, stats, mode=f'cv-{fold}')
    return test_result


def initialize_model(train_exs, dev_exs, test_exs):
    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    logger.info('Training model from scratch...')
    model = init_from_scratch(args, train_exs, dev_exs, test_exs)
    # Set up optimizer
    model.init_optimizer()

    # Use the GPU?
    device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")
    model.to(device)

    # Use multiple GPUs?
    if args.parallel:
        model.parallelize()
    return model

# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(args):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load data files')
    train_exs = utils.load_data(args.train_file)
    logger.info(f'Num train examples = {len(train_exs)}')
    if args.dev_file:
        dev_exs = utils.load_data(args.dev_file)
        logger.info(f'Num dev examples = {len(dev_exs)}')
    else:
        dev_exs = []
        logger.info('No dev data. Randomly choose 10% of train data to validate.')
    if args.test_file:
        test_exs = utils.load_data(args.test_file)
        logger.info(f'Num test examples = {len(test_exs)}')
    else:
        test_exs = []
        logger.info('No test data. Use 10 fold cv to evaluate.')
    logger.info(f'Total {len(train_exs) + len(dev_exs) + len(test_exs)} examples.')

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    logger.info('-' * 100)
    logger.info('Make data loaders')
    if args.test_file:
        model = initialize_model(train_exs, dev_exs, test_exs)
        train_loader, dev_loader, test_loader = utils.split_loader(train_exs, test_exs, args, model,
                                                                   dev_exs=dev_exs)
        result = train_valid_loop(train_loader, dev_loader, args, model, test_loader=test_loader)[args.valid_metric]
        logger.info('-' * 100)
        logger.info(f'Test {args.valid_metric}: {result*100:.2f}%')
    else:
        # 10-cross cv
        results = []
        samples_fold = [np.random.randint(10) for _ in range(len(train_exs))]
        fold_samples = defaultdict(list)
        for sample_idx, sample_fold in enumerate(samples_fold):
            fold_samples[sample_fold].append(sample_idx)
        for fold in range(10):
            fold_info = f'for fold {fold}' if fold is not None else ''
            print(colored(f'\nStarting training {fold_info}...\n', 'blue'))
            model = initialize_model(train_exs, dev_exs, test_exs)
            train_loader, dev_loader = utils.split_loader_cv(train_exs, args, model, fold_samples[fold])
            result = train_valid_loop(train_loader, dev_loader, args, model, fold=fold)
            results.append(result[args.valid_metric])
        result = np.mean(results).item()
        logger.info('-' * 100)
        logger.info(f'CV {args.valid_metric}: {result*100:.2f}%')


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'Sentence Classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # Set cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Set random state
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    # this will **slower** the speed a lot, but enforce deterministic result for CNN model
    # torch.backends.cudnn.enabled = False

    # Set logging
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)
