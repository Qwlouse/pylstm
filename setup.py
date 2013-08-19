#!/usr/bin/python
# coding=utf-8

from distutils.core import setup


setup(
    name="PyLSTM",
    version="0.2",
    packages=['pylstm', 'pylstm.wrapper', 'pylstm.structure', 'pylstm.datasets',
              'pylstm.regularization', 'pylstm.training'],
    package_data={'pylstm.wrapper': ['*.so']},
    requires=['numpy', 'scipy'],
)
