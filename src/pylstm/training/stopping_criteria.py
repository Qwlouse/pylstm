#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
from .monitoring import Monitor


class ErrorRises(Monitor):
    __default_values__ = {'delay': 1}

    def __init__(self, error, delay=1, name=None):
        super(ErrorRises, self).__init__(name, 'epoch', 1)
        self.error = error.split('.')
        self.delay = delay

    def __call__(self, epoch, net, stepper, logs):
        errors = logs
        for en in self.error:
            errors = errors[en]
        best_error_idx = np.argmin(errors)
        if len(errors) > best_error_idx + self.delay:
            raise StopIteration("Error did not fall for %d epochs! Stopping."
                                % self.delay)


class MaxEpochsSeen(Monitor):
    def __init__(self, max_epochs, name=None):
        super(MaxEpochsSeen, self).__init__(name, 'epoch', 1)
        self.max_epochs = max_epochs

    def __call__(self, epoch, net, stepper, logs):
        if epoch >= self.max_epochs:
            raise StopIteration("Max epochs reached.")