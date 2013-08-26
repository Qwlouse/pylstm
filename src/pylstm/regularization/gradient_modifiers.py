#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np


class L1(object):
    """
    L1-norm weight regularization. Schould be added to the network via the
    set_regularizers method like so:
    >> net.set_regularizers(RnnLayer=L1())
    See Network.set_regularizers for more information on how to control which
    weights to affect.
    """
    def __init__(self, reg_coeff=0.01):
        self.reg_coeff = reg_coeff
        
    def __call__(self, view, grad):
        return grad + self.reg_coeff*np.sign(view)
        

class L2(object):
    """
    L2-norm weight regularization (aka Weight-Decay). Schould be added to the
    network via the set_regularizers method like so:
    >> net.set_regularizers(RnnLayer=L2())
    See Network.set_regularizers for more information on how to control which
    weights to affect.
    """
    def __init__(self, reg_coeff=0.01):
        self.reg_coeff = reg_coeff
        
    def __call__(self, view, grad):
        return grad + self.reg_coeff*view


class GradientClipping(object):
    """
    Clips (limits) the gradient to be between low and high.
    Defaults to low=-1 and high=1.

    >> net.set_regularizers(RnnLayer=GradientClipping(-2, 2))
    See Network.set_regularizers for more information on how to control which
    weights to affect.
    """
    def __init__(self, low=-1, high=None):
        self.low, self.high = sorted([low,  high if high is not None else -low])

    def __call__(self, view, grad):
        return np.clip(grad, self.low, self.high)

