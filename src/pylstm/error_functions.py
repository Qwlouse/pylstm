#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np


class ErrorFunction(object):
    def evaluate(self, Y, T):
        pass

    def deriv(self, Y, T):
        pass


class MeanSquaredError(ErrorFunction):
    def evaluate(self, Y, T):
        Y = Y.as_array()
        return 0.5 * np.sum((Y - T) ** 2)

    def deriv(self, Y, T):
        Y = Y.as_array()
        return Y - T

