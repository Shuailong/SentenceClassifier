#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

DEFAULTS = {
    'corenlp_classpath': os.getenv('CLASSPATH')
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


from .corenlp_tokenizer import CoreNLPTokenizer


def get_class(name):
    if name == 'corenlp':
        return CoreNLPTokenizer

    raise RuntimeError('Invalid tokenizer: %s' % name)

