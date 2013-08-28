#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
import time
from .callbacks import print_error_per_epoch
from .train_steps import SGDStep, ForwardStep


class Trainer(object):
    def __init__(self, net, core=None, **kwargs):
        self.net = net
        self.stepper = core if core else SGDStep(**kwargs)
        self.validation_stepper = ForwardStep()
        self.stopping_criteria = []
        self.callbacks = [print_error_per_epoch]
        self.training_errors = []
        self.validation_errors = []
        self.epochs_seen = 0

    def emit_callbacks(self):
        for cb in self.callbacks:
            cb(self.epochs_seen, self.net,
               self.training_errors, self.validation_errors)

    def should_stop(self):
        for sc in self.stopping_criteria:
            if sc(self.epochs_seen, self.net,
                  self.training_errors, self.validation_errors):
                return True
        return False

    def restart_stopping_criteria(self):
        for sc in self.stopping_criteria:
            try:
                sc.restart()
            except AttributeError:
                pass

    def train(self, training_data_getter, validation_data_getter=None):
        # May add a default MaxEpochsSeen here if that feels better to the soul
        self.stepper.start(self.net)
        self.validation_stepper.start(self.net)
        self.restart_stopping_criteria()

        while True:
            train_errors = []
            print("\nTraining ...")
            start = time.time()
            for x, t, m in training_data_getter():
                train_errors.append(self.stepper.run(x, t, m))
            print("Wall Time taken: ", time.time() - start)

            train_error = np.mean(train_errors)
            self.training_errors.append(train_error)

            if validation_data_getter is not None:
                valid_errors = []
                print("Validating ...")
                start = time.time()
                for x, t, m in validation_data_getter():
                    valid_errors.append(self.validation_stepper.run(x, t, m))
                print("Wall Time taken: ", time.time() - start)

                valid_error = np.mean(valid_errors)
                self.validation_errors.append(valid_error)

            self.epochs_seen += 1

            self.emit_callbacks()

            if self.should_stop():
                return train_error
