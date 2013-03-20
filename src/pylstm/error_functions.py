#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from pylstm.wrapper import Buffer


def ensure_np_array(a):
    if isinstance(a, Buffer):
        return a.as_array()
    else:
        return np.array(a)


class ErrorFunction(object):
    def evaluate(self, Y, T):
        pass

    def deriv(self, Y, T):
        pass


class MeanSquaredError(ErrorFunction):
    def evaluate(self, Y, T):
        Y = ensure_np_array(Y)
        return 0.5 * np.sum((Y - T) ** 2)

    def deriv(self, Y, T):
        Y = ensure_np_array(Y)
        return Y - T


class CrossEntropyError(object):
    def __call__(self, Y, T):
        Y = ensure_np_array(Y)
        Y[Y < 1e-6] = 1e-6
        cee = T * np.log(Y)
        return - np.sum(cee)

    def evaluate(self, Y, T):
        return self(Y, T)

    def deriv(self, Y, T):
        Y = ensure_np_array(Y)
        return - T / Y

