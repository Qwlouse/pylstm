#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
import sys
import numpy as np
from pylstm.describable import Describable


class Trainer(Describable):
    __undescribed__ = {
        'current_epoch': 0,
        'logs': {}
    }
    __default_values__ = {'verbose': True}

    def __init__(self, stepper, verbose=True):
        self.stepper = stepper
        self.verbose = verbose
        self.monitors = OrderedDict()
        self.current_epoch = 0
        self.logs = dict()

    def __init_from_description__(self, description):
        # recover the order of the monitors from their priorities
        # and set their names
        ordered_mon = sorted(self.monitors.items(), key=_get_priority)
        self.monitors = OrderedDict()
        for name, mon in ordered_mon:
            self.monitors[name] = mon
            mon.__name__ = name

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

    def start_monitors(self, net, monitor_kwargs):
        self.logs = {'training_errors': [float('NaN')]}
        for name, monitor in self.monitors.items():
            try:
                if hasattr(monitor, 'start'):
                    monitor.start(net, self.stepper, self.verbose,
                                  monitor_kwargs)
            except Exception as err:
                print(err)
                pass

    def _add_log(self, name, val, logs=None, indent=0):
        if logs is None:
            logs = self.logs
        if isinstance(val, dict):
            if self.verbose:
                print(" " * indent + name)
            if name not in logs:
                logs[name] = dict()
            for k, v in val.items():
                self._add_log(k, v, logs[name], indent+2)
        elif val is not None:
            if self.verbose:
                print(" " * indent + ("{0:%d}: {1}" % (40-indent)).format(name, val))
            if name not in logs:
                logs[name] = []
            logs[name].append(val)

    def emit_monitoring(self, net, timescale, update_nr=None):
        monitoring_arguments = dict(
            epoch=self.current_epoch,
            net=net,
            stepper=self.stepper,
            logs=self.logs
        )
        update_nr = self.current_epoch if timescale == 'epoch' else update_nr
        should_stop = False
        for name, monitor in self.monitors.items():
            m_timescale, interval = _get_monitor_params(monitor)
            if m_timescale != timescale:
                continue
            if update_nr % interval == 0:
                monitor_log, stop = _call_monitor(monitor, monitoring_arguments)
                should_stop |= stop
                self._add_log(name, monitor_log)

        return should_stop

    def train(self, net, training_data_getter, **monitor_kwargs):
        if self.verbose:
            print('\n\n', 15 * '- ', "Pretraining", 15 * ' -')
        self.stepper.start(net)
        self.start_monitors(net, monitor_kwargs)
        self.emit_monitoring(net, 'epoch')
        while True:
            self.current_epoch += 1
            sys.stdout.flush()
            train_errors = []
            if self.verbose:
                print('\n\n', 15 * '- ', "Epoch", self.current_epoch, 15 * ' -')
            for i, (x, t) in enumerate(
                    training_data_getter(verbose=self.verbose)):
                train_errors.append(self.stepper.run(x, t))
                if self.emit_monitoring(net, 'update', i + 1):
                    break

            self._add_log('training_errors', np.mean(train_errors))
            if self.emit_monitoring(net, 'epoch'):
                break


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


def _call_monitor(monitor, monitoring_arguments):
    try:
        return monitor(**monitoring_arguments), False
    except StopIteration as err:
        print(">> Stopping because:", err)
        if hasattr(err, 'value'):
            return err.value, True

        return None, True
    except Exception as err:
        err.args = (err.args[0] + " in " + str(monitor),)
        raise