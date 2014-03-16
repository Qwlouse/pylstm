#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
from pylstm.randomness import Seedable


class Constraint(object):
    def __get_description__(self):
        description = self.__dict__.items()
        description['$type'] = self.__class__.__name__
        return description

    def __init_from_description__(self, description):
        assert self.__class__.__name__ == description['$type']
        self.__dict__.update({k: v for k, v in description.items()
                              if k != '$type'})

    def __call__(self, view):
        raise NotImplementedError()


def get_constraints_description(constraint):
    """
    Turn a constraints-dictionary as used in the Network.set_constraints method
    into a description dictionary. This description is json serializable.

    :param constraint: constraints-dictionary
    :type constraint: dict
    :return: description
    :rtype: dict
    """
    if isinstance(constraint, Constraint):
        return constraint.__get_description__()
    elif isinstance(constraint, dict):
        return {k: get_constraints_description(v)
                for k, v in constraint.items()}
    else:
        return constraint


class RescaleIncomingWeights(Constraint):
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

    def __repr__(self):
        return "<RescaleIncomingWeights %0.4f>" % self.target_sum


class LimitIncomingWeightsSquared(Constraint):
    """
    Limits the squares of incoming weights for every neuron to sum to one
    (target_sum). Ignores Biases.

    Should be added to the network via the set_constraints method like so:
    >> net.set_constraints(RnnLayer={'HX': LimitIncomingWeightsSquared()})
    See Network.set_constraints for more information on how to control which
    weights to affect.
    """
    def __init__(self, target_sum=1.0):
        self.target_sum = target_sum

    def __call__(self, view):
        if view.shape[1] == 1:  # Just one input: probably bias => ignore
            return view
        sums = (view*view).sum(1)
        sums = (sums < self.target_sum) + \
               (sums / self.target_sum) * (sums >= self.target_sum)
        return view / np.sqrt(sums)

    def __repr__(self):
        return "<LimitIncomingWeightsSquared %0.4f>" % self.target_sum


class ClipWeights(Constraint):
    """
    Clips (limits) the weights to be between low and high.
    Defaults to low=-1 and high=1.

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

    def __repr__(self):
        return "<ClipWeights [%0.4f; %0.4f]>" % (self.low, self.high)


class MaskWeights(Constraint):
    """
    Multiplies the weights with the mask. This can be used to clamp some of
    the weights to zero.

    Should be added to the network via the set_constraints method like so:
    >> net.set_constraints(RnnLayer={'HR': MaskWeights(M)})
    See Network.set_constraints for more information on how to control which
    weights to affect.
    """
    def __init__(self, mask):
        assert isinstance(mask, np.ndarray)
        self.mask = mask

    def __call__(self, view):
        return view * self.mask

    def __repr__(self):
        return "<MaskWeights>"

    def __get_description__(self):
        return {
            '$type': self.__class__.__name__,
            'mask': self.mask.tolist()
        }

    def __init_from_description__(self, description):
        assert self.__class__.__name__ == description['$type']
        self.mask = np.array(description['mask'])


class FreezeWeights(Constraint):
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

    def __repr__(self):
        return "<FreezeWeights>"

    def __get_description__(self):
        return {'$type': self.__class__.__name__}

    def __init_from_description__(self, description):
        assert self.__class__.__name__ == description['$type']
        self.weights = None


class NoisyWeights(Seedable, Constraint):
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

    def __init__(self, std=0.01):
        super(NoisyWeights, self).__init__()
        self.std = std
        self.noise = None

    def __call__(self, view):
        if self.noise is None:
            self.noise = np.zeros_like(view)
        old_noise = self.noise
        self.noise = self.rnd.randn(*view.shape) * self.std
        return view - old_noise + self.noise

    def __repr__(self):
        return "<NoisyWeights std=%0.4f>" % self.std

    def __get_description__(self):
        return {
            '$type': self.__class__.__name__,
            'std': self.std
        }

    def __init_from_description__(self, description):
        assert self.__class__.__name__ == description['$type']
        self.std = description['std']
        self.noise = None