#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
import sys
import time
import numpy as np
from .train_steps import SgdStep, ForwardStep
from pylstm.describable import Describable


class Trainer(Describable):
    __undescribed__ = {
        'validation_stepper': ForwardStep(),
        'training_errors': [],
        'validation_errors': [],
        'epochs_seen': 0
    }

    def __init__(self, stepper=None, verbose=True, **kwargs):
        self.stepper = stepper if stepper else SgdStep(**kwargs)
        self.verbose = verbose
        self.validation_stepper = ForwardStep()
        self.stopping_criteria = []
        self.training_errors = []
        self.validation_errors = []
        self.monitor = OrderedDict()
        self.epochs_seen = 0

    def emit_monitoring_batchwise(self, update_nr, net):
        monitoring_arguments = dict(
            epoch=self.epochs_seen,
            net=net,
            stepper=self.stepper,
            training_errors=self.training_errors,
            validation_errors=self.validation_errors
        )
        for mon in self.monitor.values():
            timescale, interval = _get_monitor_params(mon)
            if timescale == 'epoch':
                continue
            if update_nr % interval == 0:
                mon(**monitoring_arguments)

    def emit_monitoring_epochwise(self, net):
        monitoring_arguments = dict(
            epoch=self.epochs_seen,
            net=net,
            stepper=self.stepper,
            training_errors=self.training_errors,
            validation_errors=self.validation_errors
        )
        for mon in self.monitor.values():
            timescale, interval = _get_monitor_params(mon)
            if timescale == 'update':
                continue
            if self.epochs_seen % interval == 0:
                mon(**monitoring_arguments)

    def should_stop(self, net):
        for sc in self.stopping_criteria:
            if sc(self.epochs_seen, net,
                  self.training_errors, self.validation_errors):
                return True
        return False

    def restart_stopping_criteria(self):
        for sc in self.stopping_criteria:
            try:
                sc.restart()
            except AttributeError:
                pass

    def train(self, net, training_data_getter, validation_data_getter=None):
        self.stepper.start(net)
        self.validation_stepper.start(net)
        self.restart_stopping_criteria()

        while True:
            sys.stdout.flush()
            train_errors = []
            if self.verbose:
                print('\n\n', 15*'- ', " Epoch ", (self.epochs_seen + 1),
                      15 * ' -')
                print("Training ...")
            start = time.time()
            for i, (x, t) in enumerate(training_data_getter()):
                train_errors.append(self.stepper.run(x, t))
                self.emit_monitoring_batchwise(i + 1, net)

            if self.verbose:
                print("Wall Time taken: ", time.time() - start)

            train_error = np.mean(train_errors)
            self.training_errors.append(train_error)

            if validation_data_getter is not None:
                valid_errors = []
                if self.verbose:
                    print("Validating ...")
                start = time.time()
                for x, t in validation_data_getter():
                    valid_errors.append(self.validation_stepper.run(x, t))
                if self.verbose:
                    print("Wall Time taken: ", time.time() - start)

                valid_error = np.mean(valid_errors)
                self.validation_errors.append(valid_error)

            self.epochs_seen += 1
            self.emit_monitoring_epochwise(net)
            if self.should_stop(net):
                return train_error


def _get_monitor_params(monitor):
    timescale = monitor.timescale if hasattr(monitor, 'timescale') else 'epoch'
    interval = monitor.interval if hasattr(monitor, 'interval') else 1
    return timescale, interval