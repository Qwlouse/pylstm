#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from copy import copy
import numpy as np


############################ Base Class ########################################

class Regularizer(object):
    def __get_description__(self):
        description = copy(self.__dict__)
        description['$type'] = self.__class__.__name__
        return description

    def __init_from_description__(self, description):
        assert self.__class__.__name__ == description['$type']
        self.__dict__.update({k: v for k, v in description.items()
                              if k != '$type'})


############################ Regularizers ######################################

class L1(Regularizer):
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

    def __repr__(self):
        return "<L1 %0.4f>" % self.reg_coeff
        

class L2(Regularizer):
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

    def __repr__(self):
        return "<L2 %0.4f>" % self.reg_coeff


class ClipGradient(Regularizer):
    """
    Clips (limits) the gradient to be between low and high.
    Defaults to low=-1 and high=1.

    >> net.set_regularizers(RnnLayer=ClipGradient(-2, 2))
    See Network.set_regularizers for more information on how to control which
    weights to affect.
    """
    def __init__(self, low=-1, high=None):
        high = high if high is not None else -low
        self.low, self.high = sorted([low,  high])

    def __call__(self, view, grad):
        return np.clip(grad, self.low, self.high)

    def __repr__(self):
        return "<ClipGradient [%0.4f; %0.4f]>" % (self.low, self.high)


############################ helper methods ####################################

def _get_regularization_description(regularizer):
    """
    Turn a regularization-dictionary as used in the Network.set_regularizers
    method into a description dictionary. This description is json serializable.

    :param regularizer: regularization-dictionary
    :type regularizer: dict
    :return: description
    :rtype: dict
    """
    if isinstance(regularizer, Regularizer):
        return regularizer.__get_description__()
    elif isinstance(regularizer, dict):
        return {k: _get_regularization_description(v)
                for k, v in regularizer.items()}
    else:
        return regularizer


def _create_regularizer_from_description(description):
    """
    Turn an regularization-description into a regularization dictionary, that
    can be used with Network.set_regularizers.

    :param description: regularization-description
    :type description: dict

    :return: regularization-dictionary
    :rtype: dict
    """
    if isinstance(description, dict):
        if '$type' in description:
            name = description['$type']
            for initializer in Regularizer.__subclasses__():
                if initializer.__name__ == name:
                    instance = initializer.__new__(initializer)
                    instance.__init_from_description__(description)
                    return instance
            raise RuntimeError('Regularizer "%s" not found!' % name)
        else:
            return {k: _create_regularizer_from_description(v)
                    for k, v in description.items()}
    else:
        raise RuntimeError('illegal description type "%s"' %
                           type(description))
