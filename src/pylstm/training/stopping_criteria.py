#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np


class ValidationErrorRises(object):
    def __init__(self, delay=1):
        self.delay = delay

    def restart(self):
        pass

    def __call__(self, epochs_seen, net, training_errors, validation_errors):
        best_val_error = np.argmin(validation_errors)
        if len(validation_errors) > best_val_error + self.delay:
            print("Validation error did not fall for %d epochs! Stopping."
                  % self.delay)
            return True
        return False


class MaxEpochsSeen(object):
    def __init__(self, max_epochs=100):
        self.max_epochs = max_epochs

    def restart(self):
        pass

    def __call__(self, epochs_seen, net, training_errors, validation_errors):
        if epochs_seen >= self.max_epochs:
            return True