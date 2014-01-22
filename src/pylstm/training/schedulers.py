#!/usr/bin/python
# coding=utf-8
"""
Quantities like learning_rate and momentum in SgdStep, NesterovStep, MomentumStep etc.
can be changed according to schedules instead of being constants.
These common schedulers are provided for convenience.
"""
from __future__ import division, print_function, unicode_literals
import numpy as np


class LinearSchedule(object):
    """
    Change the quantity linearly from 'initial_value' to 'final_value'
    by changing the quantity every 'interval' number of updates.
    Step size is decided by num_changes.
    For example:
    LinearSchedule(0.9, 1.0, 10, 100)
    will change the quantity starting from 0.9 to 1.0 by incrementing it
    after every 100 updates such that 10 increments are made.
    """
    def __init__(self, initial_value=0, final_value=0, num_changes=1, interval=1, name=""):
        self.initial_value = initial_value
        self.final_value = final_value
        self.interval = interval
        self.num_changes = num_changes
        self.update_number = 0  # initial_value should be used for first update
        self.current_value = None
        self.name = name

    def __call__(self):
        if (self.update_number // self.interval) > self.num_changes:
            self.current_value = self.final_value
        else:
            self.current_value = self.initial_value + ((self.final_value - self.initial_value) / self.num_changes) * \
                                 ((self.update_number // self.interval))
        self.update_number += 1
        return self.current_value


class ExponentialSchedule(object):
    def __init__(self, initial_value=0, factor=1, interval=1, minimum=-np.Inf, maximum=np.Inf, name=""):
        self.initial_value = initial_value
        self.factor = factor
        self.interval = interval
        self.minimum = minimum
        self.maximum = maximum
        self.update_number = 0
        self.current_value = None
        self.name = name

    def __call__(self):
        self.current_value = min(
            max(self.minimum, self.initial_value * (self.factor ** ((self.update_number // self.interval)))),
            self.maximum)
        self.update_number += 1
        return self.current_value

