#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
from pylstm.describable import Describable


class StoppingCriterion(Describable):
    __undescribed__ = {'__name__'}

    def __init__(self, name=None):
        if name is None:
            self.__name__ = self.__class__.__name__
        else:
            self.__name__ = name

    def __call__(self, epochs_seen, net, training_errors, validation_errors):
        raise NotImplementedError()

    def restart(self):
        pass


class ValidationErrorRises(StoppingCriterion):
    def __init__(self, delay=1, name=None):
        super(ValidationErrorRises, self).__init__(name)
        self.delay = delay

    def __call__(self, epochs_seen, net, training_errors, validation_errors):
        assert validation_errors, "You need to specify a validation set " \
                                  "if you want to use ValidationErrorRises"
        best_val_error = np.argmin(validation_errors)
        if len(validation_errors) > best_val_error + self.delay:
            print("Validation error did not fall for %d epochs! Stopping."
                  % self.delay)
            return True
        return False


class MaxEpochsSeen(StoppingCriterion):
    def __init__(self, max_epochs=100, name=None):
        super(MaxEpochsSeen, self).__init__(name)
        self.max_epochs = max_epochs

    def __call__(self, epochs_seen, net, training_errors, validation_errors):
        if epochs_seen >= self.max_epochs:
            return True