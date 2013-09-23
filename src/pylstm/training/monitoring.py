#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np


def MonitorFunction(timescale='epoch', interval=1):
    """
    Decorator to adjust the interval and timescale for a monitoring function.
    Example:

    @MonitorFunction('update', 5)
    def foo(*args, **kwargs):
        print('bar')
    """
    def decorator(f):
        f.timescale = timescale
        f.interval = interval
        return f
    return decorator


def print_error_per_epoch(epoch, training_errors, validation_errors, **_):
    if len(validation_errors) == 0:
        print("\nEpoch %d:\tTraining error = %0.4f" % (epoch,
                                                       training_errors[-1]))
    else:
        print("\nEpoch %d:\tTraining error = %0.4f Validation error = %0.4f" %
              (epoch, training_errors[-1], validation_errors[-1]))


class SaveWeights(object):
    """
    Save the weights of the network to the given file on every call.
    Default is to save them once per epoch, but this can be configured using
    the timescale and interval parameters.
    """
    def __init__(self, filename, timescale='epoch', interval=1):
        self.timescale = timescale
        self.interval = interval
        self.filename = filename

    def __call__(self, epoch, net, **_):
        np.save(self.filename.format(epoch=epoch), net.param_buffer)


class SaveBestWeights(object):
    """
    Check every epoch to see if the validation error (or training error if there
    is no validation error) is at it's minimum and if so, save the weights to
    the specified file.
    """
    def __init__(self, filename):
        self.timescale = 'epoch'
        self.interval = 1
        self.filename = filename

    def __call__(self, epoch, net, training_errors, validation_errors, **_):
        e = validation_errors if len(validation_errors) > 0 else training_errors
        if np.argmin(e) == len(e) - 1:
            filename = self.filename.format(epoch=epoch)
            print("Saving weights to {0}...".format(filename))
            np.save(filename, net.param_buffer)


class MonitorClassificationError(object):
    """
    Monitor the classification error assuming one-hot encoding of targets.
    """
    def __init__(self, X, T, M=None, name="", timescale='epoch', interval=1):
        self.timescale = timescale
        self.interval = interval
        self.X = X
        self.T = T
        self.M = M
        self.name = name
        self.log = dict()
        self.log['classification_error'] = []

    def __call__(self, epoch, net, training_errors, validation_errors, **_):
        print("\nEvaluating classification error for channel: ", self.name)
        Y = net.forward_pass(self.X)
        Y_win = Y.argmax(2).reshape(self.T.shape[0], self.T.shape[1], 1)
        T_win = self.T.argmax(2).reshape(self.T.shape[0], self.T.shape[1], 1)
        total_errors = np.sum((Y_win != T_win) * self.M)
        total = np.sum(self.M)
        error_fraction = total_errors / total
        self.log['classification_error'].append(error_fraction)
        print(self.name, ":\tClassificiation Error = ", error_fraction)