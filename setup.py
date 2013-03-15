#!/usr/bin/python
# coding=utf-8

from distutils.core import setup


setup(
    name="PyLSTM",
    version="0.1",
    packages=['pylstm', 'pylstm.wrapper'],
    package_data={'pylstm.wrapper': ['*.so']},
)
