#!/usr/bin/env python
# coding=utf-8

from distutils.core import setup


setup(
    name="pylstm",
    version="0.4",
    packages=['pylstm', 'pylstm.wrapper', 'pylstm.structure', 'pylstm.datasets',
              'pylstm.regularization', 'pylstm.training'],
    package_data={'pylstm.wrapper': ['*.so']},
    requires=['numpy'],
)
