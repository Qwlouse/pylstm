#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np


class RescaleIncomingWeights(object):
    """
    Rescales the incoming weights for every neuron to sum to one (target_sum).
    Ignores Biases.

    Should be added to the network via the set_constraints method like so:
    >> net.set_constraints(RnnLayer={'HX': RescaleIncomingWeights()})
    See Network.set_constraints for more information on how to control which
    weights to affect.
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
    Defaults to min_value=-1 and high=1.

    Should be added to the network via the set_constraints method like so:
    >> net.set_constraints(RnnLayer={'HR': ClipWeights()})
    See Network.set_constraints for more information on how to control which
    weights to affect.
    """
    def __init__(self, low=-1., high=1.):
        self.low = low
        self.high = high

    def __call__(self, view):
        return np.clip(view, self.low, self.high)


class MaskWeights(object):
    """
    Multiplies the weights with the mask. This can be used to clamp some of
    the weights to zero.

    Should be added to the network via the set_constraints method like so:
    >> net.set_constraints(RnnLayer={'HR': MaskWeights(M)})
    See Network.set_constraints for more information on how to control which
    weights to affect.
    """
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, view):
        return view * self.mask


class FreezeWeights(object):
    """
    Prevents the weights from changing at all. So it will remember the first
    weights it sees and resets them to that every time. This means it should
    typically be added to the network BEFORE initialization.

    Should be added to the network via the set_constraints method like so:
    >> net.set_constraints(RnnLayer={'HR': FreezeWeights()})
    See Network.set_constraints for more information on how to control which
    weights to affect.
    """
    def __init__(self):
        self.weights = None

    def __call__(self, view):
        if self.weights is None:
            self.weights = view
        return self.weights


class NoisyWeights(object):
    """
    Adds a small amount of normal-distributed noise (mean=0, std=std) to all the
    weights of a network every time they are set. This means that you get
    different weights for every batch. So if you want noisy weights per sequence
    you need to use it together with online training.

    Should be added to the network via the set_constraints method like so:
    >> net.set_constraints(RnnLayer={'HR': NoisyWeights()})
    See Network.set_constraints for more information on how to control which
    weights to affect.
    """
    def __init__(self, std=0.01, seed=None):
        self.rnd = np.random.RandomState(seed)
        self.std = 0.01

    def __call__(self, view):
        size = reduce(np.multiply, view.shape)
        return (view.flatten() - self.rnd.randn(size) * self.std).reshape(*view.shape)