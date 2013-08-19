#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np


def print_error_per_epoch(epoch, net, training_errors, validation_errors):
    if len(validation_errors) == 0:
        print("\nEpoch %d:\tTraining error = %0.4f" % (epoch,
                                                       training_errors[-1]))
    else:
        print("\nEpoch %d:\tTraining error = %0.4f Validation error = %0.4f" %
              (epoch, training_errors[-1], validation_errors[-1]))


class SaveWeightsPerEpoch(object):
    def __init__(self, filename):
        self.filename = filename

    def __call__(self, epoch, net, training_errors, validation_errors):
        np.save(self.filename.format(epoch=epoch), net.param_buffer)


class SaveBestWeights(object):
    def __init__(self, filename):
        self.filename = filename

    def __call__(self, epoch, net, training_errors, validation_errors):
        e = validation_errors if len(validation_errors) > 0 else training_errors
        if np.argmin(e) == len(e) - 1:
            filename = self.filename.format(epoch=epoch)
            print("Saving weights to {0}...".format(filename))
            np.save(filename, net.param_buffer)
