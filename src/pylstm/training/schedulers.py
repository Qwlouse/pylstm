#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np


class LinearSchedule(object):
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

