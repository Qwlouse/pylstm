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


class CrossEntropyError(object):
    def __call__(self, Y, T):
        Y = Y.as_array()
        Y[Y < 1e-6] = 1e-6
        cee = T * np.log(Y)
        return - np.sum(cee)

    def evaluate(self, Y, T):
        return self(Y, T)

    def deriv(self, Y, T):
        Y = Y.as_array()
        return T / Y

