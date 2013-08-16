#!/usr/bin/python
# coding=utf-8

from distutils.core import setup


setup(
    name="PyLSTM",
    version="0.2",
    packages=['pylstm', 'pylstm.wrapper', 'pylstm.layers'],
    package_data={'pylstm.wrapper': ['*.so']},
    requires=['numpy', 'scipy'],
)
