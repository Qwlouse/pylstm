#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np


class RescaleIncomingWeights(object):
    """
    Rescales the incoming weights for every neuron to sum to one (target_sum).
    Ignores Biases.
    """
    def __init__(self, target_sum=1.0):
        self.target_sum = target_sum

    def __call__(self, view):
        if view.shape[1] == 1:  # Just one input: probably bias => ignore
            return view
        return view / view.sum(1) * self.target_sum


class ClipWeights(object):
    """
    Clips (limits) the weights to be between low and high.
    Defaults to min_value=-1 and high=1
    """
    def __init__(self, low=-1., high=1.):
        self.low = low
        self.high = high

    def __call__(self, view):
        return np.clip(view, self.low, self.high)