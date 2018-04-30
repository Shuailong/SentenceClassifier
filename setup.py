#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# Adapt from facebookresearch/DrQA by Shuailong on Mar 22 2018.

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    reqs = f.read()


setup(
    name='Sentence Classifier',
    version='0.1.0',
    description='PyTorch based Sentence Classifier',
    author='Liang Shuailong',
    author_email='liangshuailong@gmail.com',
    long_description=readme,
    license=license,
    python_requires='>=3.6',
    packages=find_packages(exclude=('data')),
    install_requires=reqs.strip().split('\n'),

)
