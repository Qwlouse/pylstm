#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from pylstm.describable import Describable
from pylstm.error_functions import ClassificationError, LabelingError, \
    get_error_function_by_name
from collections import OrderedDict


class Monitor(Describable):
    __undescribed__ = {'__name__'}  # the name is saved in the trainer
    __default_values__ = {
        'timescale': 'epoch',
        'interval': 1,
        'verbose': None
    }

    def __init__(self, name=None, timescale='epoch', interval=1, verbose=None):
        self.timescale = timescale
        self.interval = interval
        if name is None:
            self.__name__ = self.__class__.__name__
        else:
            self.__name__ = name
        self.priority = 0
        self.verbose = verbose

    def start(self, net, stepper, verbose, monitor_kwargs):
        if self.verbose is None:
            self.verbose = verbose

    def __call__(self, epoch, net, stepper, logs):
        pass


class SaveWeights(Monitor):
    """
    Save the weights of the network to the given file on every call.
    Default is to save them once per epoch, but this can be configured using
    the timescale and interval parameters.
    """

    def __init__(self, filename, name=None, timescale='epoch', interval=1):
        super(SaveWeights, self).__init__(name, timescale, interval)
        self.filename = filename

    def __call__(self, epoch, net, stepper, logs):
        np.save(self.filename, net.param_buffer)

    def load_weights(self):
        return np.load(self.filename)


class SaveBestWeights(Monitor):
    """
    Check every epoch to see if the validation error (or training error if there
    is no validation error) is at it's minimum and if so, save the weights to
    the specified file.
    """
    __undescribed__ = {'weights': None}
    __default_values__ = {'filename': None}

    def __init__(self, error_log_name, filename=None, name=None, verbose=None):
        super(SaveBestWeights, self).__init__(name, 'epoch', 1, verbose)
        self.error_log_name = error_log_name.split('.')
        self.filename = filename
        self.weights = None

    def __call__(self, epoch, net, stepper, logs):
        e = logs
        for en in self.error_log_name:
            e = e[en]
        min_error_idx = np.argmin(e)
        if min_error_idx == len(e) - 1:
            if self.filename is not None:
                if self.verbose:
                    print(">> Saving weights to {0}...".format(self.filename))
                np.save(self.filename, net.param_buffer)
            else:
                if self.verbose:
                    print(">> Caching weights")
                self.weights = net.param_buffer.copy()
        elif self.verbose:
            print(">> Last saved weigths after epoch {}".format(min_error_idx))

    def load_weights(self):
        return np.load(self.filename) if self.filename is not None \
            else self.weights


class AggregateMeanError(Describable):
    def __call__(self, x):
        return np.mean(x, axis=0)

aggregate_mean_error = AggregateMeanError()


class AggregateClassError(Describable):
    def __call__(self, x):
        e = np.sum(x, axis=0)
        return np.round(e[0] * 100. / e[1], 2)

aggregate_class_error = AggregateClassError()


class MonitorError(Monitor):
    """
    Monitor the given error (aggregated over all sequences).
    """
    __undescribed__ = {'data_iter'}
    __default_values__ = {'aggregator': aggregate_mean_error}

    def __init__(self, data_name, error_func=None,
                 aggregator=aggregate_mean_error,
                 name=None, timescale='epoch', interval=1):
        if name is None and error_func is not None:
            name = 'Monitor' + error_func.__name__
        super(MonitorError, self).__init__(name, timescale, interval)
        assert isinstance(data_name, basestring)
        self.data_name = data_name
        self.data_iter = None
        self.error_func = error_func
        self.aggregator = aggregator

    def start(self, net, stepper, verbose, monitor_kwargs):
        super(MonitorError, self).start(net, stepper, verbose, monitor_kwargs)
        self.data_iter = monitor_kwargs[self.data_name]

    def __call__(self, epoch, net, stepper, logs):
        error_func = self.error_func or net.error_func
        errors = []
        for x, t in self.data_iter(self.verbose):
            y = net.forward_pass(x)
            error, _ = error_func(y, t)
            errors.append(error)
        return self.aggregator(errors)


class MonitorClassificationError(MonitorError):
    __default_values__ = {'aggregator': aggregate_class_error}

    def __init__(self, data_name, name=None, timescale='epoch', interval=1):
        super(MonitorClassificationError, self).__init__(
            data_name,
            error_func=ClassificationError,
            aggregator=aggregate_class_error,
            name=name, timescale=timescale, interval=interval)


class MonitorLabelingError(MonitorError):
    __default_values__ = {'aggregator': aggregate_class_error}

    def __init__(self, data_name, name=None, timescale='epoch', interval=1):
        super(MonitorLabelingError, self).__init__(
            data_name,
            error_func=LabelingError,
            aggregator=aggregate_class_error,
            name=name, timescale=timescale, interval=interval)


class MonitorMultipleErrors(Monitor):
    """
    Monitor errors (aggregated over all sequences).
    """
    __undescribed__ = {'data_iter', 'error_functions'}
    __default_values__ = {'aggregators': aggregate_mean_error}

    def __init__(self, data_name, error_functions,
                 aggregators=aggregate_mean_error,
                 name=None, timescale='epoch', interval=1):
        super(MonitorMultipleErrors, self).__init__(name, timescale, interval)
        self.data_name = data_name
        self.data_iter = None
        self.error_functions = error_functions
        if isinstance(aggregators, (list, tuple)):
            self.aggregators = aggregators
        else:
            self.aggregators = aggregators

    def start(self, net, stepper, verbose, monitor_kwargs):
        super(MonitorMultipleErrors, self).start(net, stepper, verbose,
                                                 monitor_kwargs)
        self.data_iter = monitor_kwargs[self.data_name]

    def __call__(self, epoch, net, stepper, logs):
        errors = {e: [] for e in self.error_functions}
        for x, t in self.data_iter(self.verbose):
            y = net.forward_pass(x)
            for error_func in self.error_functions:
                error, _ = error_func(y, t)
                errors[error_func].append(error)

        if isinstance(self.aggregators, list):
            return {err.__name__: agg(errors[err])
                    for err, agg in zip(self.error_functions, self.aggregators)}
        else:
            return {err.__name__: self.aggregators(errors[err])
                    for err in self.error_functions}

    def __describe__(self):
        description = super(MonitorMultipleErrors, self).__describe__()
        description['error_functions'] = [e.__name__
                                          for e in self.error_functions]
        return description

    def __init_from_description__(self, description):
        self.error_functions = [get_error_function_by_name(e)
                                for e in description['error_functions']]


class PlotMonitors(Monitor):
    """
    Open a window and plot the training and validation errors while training.
    """
    __undescribed__ = {'plt', 'fig', 'ax', 'lines'}

    def __init__(self, name=None, show_min=True, timescale='epoch', interval=1):
        super(PlotMonitors, self).__init__(name, timescale, interval)
        import matplotlib.pyplot as plt
        self.show_min = show_min
        self.plt = plt
        self.plt.ion()
        self.fig = None
        self.ax = None
        self.lines = None
        self.mins = None

    def start(self, net, stepper, verbose, monitor_kwargs):
        super(PlotMonitors, self).start(net, stepper, verbose, monitor_kwargs)
        self.fig, self.ax = self.plt.subplots()
        self.ax.set_title('Training Progress')
        self.ax.set_xlabel('Epochs')
        self.ax.set_ylabel('Error')
        self.lines = dict()
        self.mins = dict()
        self.plt.show()

    def _plot(self, name, data):
        data = data[1:]  # ignore pre-training entry
        x = range(1, len(data)+1)
        if name not in self.lines:
            line, = self.ax.plot(x, data, '-', label=name)
            self.lines[name] = line
        else:
            self.lines[name].set_ydata(data)
            self.lines[name].set_xdata(x)

        if not self.show_min or len(data) < 2:
            return

        min_idx = np.argmin(data) + 1
        if name not in self.mins:
            color = self.lines[name].get_color()
            self.mins[name] = self.ax.axvline(min_idx, color=color)
        else:
            self.mins[name].set_xdata(min_idx)

    def __call__(self, epoch, net, stepper, logs):
        if epoch < 2:
            return

        for name, log in logs.items():
            if not isinstance(log, (list, dict)) or not log:
                continue
            if isinstance(log, dict):
                for k, v in log.items():
                    self._plot(name + '.' + k, v)
            else:
                self._plot(name, log)

        self.ax.legend()
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.draw()  # no idea why I need that twice, but I do


class MonitorLayerProperties(Monitor):
    """
    Monitor some properties of a layer.
    """
    def __init__(self, layer_name, timescale='epoch',
                 interval=1, name=None):
        if name is None:
            name = "Monitor{}Properties".format(layer_name)
        super(MonitorLayerProperties, self).__init__(name, timescale, interval)
        self.layer_name = layer_name

    def __call__(self, epoch, net, stepper, logs):
        log = OrderedDict()
        for key, value in net.get_param_view_for(self.layer_name).items():
            log['min_' + key] = value.min()
            log['max_' + key] = value.max()
            #if key.split('_')[-1] != 'bias':
            if value.shape[1] > 1:
                log['min_sq_norm_' + key] = np.sum(value ** 2, axis=1).min()
                log['max_sq_norm_' + key] = np.sum(value ** 2, axis=1).max()
        return log