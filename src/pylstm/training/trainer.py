#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
import sys
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
        self.stopping_criteria = dict()
        self.training_errors = []
        self.validation_errors = []
        self.monitors = OrderedDict()
        self.epochs_seen = 0

    def __init_from_description__(self, description):
        # recover the order of the monitors from their priorities
        # and set their names
        ordered_mon = sorted(self.monitors.items(), key=_get_priority)
        self.monitors = OrderedDict()
        for name, mon in ordered_mon:
            self.monitors[name] = mon
            mon.__name__ = name

        # set the names of the stoppers
        for name, stopper in self.stopping_criteria.items():
            stopper.__name__ = name

    def add_monitor(self, monitor):
        for name in _name_generator(monitor.__name__):
            if name not in self.monitors:
                if self.monitors:
                    priority = self.monitors.values()[-1].priority + 1
                else:
                    priority = 0
                self.monitors[name] = monitor
                monitor.__name__ = name
                monitor.priority = priority
                break

    def add_stopper(self, stopper):
        for name in _name_generator(stopper.__name__):
            if name not in self.stopping_criteria:
                self.stopping_criteria[name] = stopper
                stopper.__name__ = name
                break

    def emit_monitoring_batchwise(self, update_nr, net):
        monitoring_arguments = dict(
            epoch=self.epochs_seen,
            net=net,
            stepper=self.stepper,
            training_errors=self.training_errors,
            validation_errors=self.validation_errors
        )
        for mon in self.monitors.values():
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
        for mon in self.monitors.values():
            timescale, interval = _get_monitor_params(mon)
            if timescale == 'update':
                continue
            if self.epochs_seen % interval == 0:
                mon(**monitoring_arguments)

    def should_stop(self, net):
        for sc in self.stopping_criteria.values():
            if sc(self.epochs_seen, net,
                  self.training_errors, self.validation_errors):
                return True
        return False

    def restart_stopping_criteria(self):
        for sc in self.stopping_criteria.values():
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
            for i, (x, t) in enumerate(training_data_getter()):
                train_errors.append(self.stepper.run(x, t))
                self.emit_monitoring_batchwise(i + 1, net)

            train_error = np.mean(train_errors)
            self.training_errors.append(train_error)

            if validation_data_getter is not None:
                valid_errors = []
                if self.verbose:
                    print("Validating ...")
                for x, t in validation_data_getter():
                    valid_errors.append(self.validation_stepper.run(x, t))

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


def _name_generator(basename):
    yield basename
    idx = 1
    while True:
        yield basename + '_' + str(idx)
        idx += 1


def _get_priority(x):
    mon = x[1]
    if hasattr(mon, 'priority'):
        return mon.priority
    else:
        return 0