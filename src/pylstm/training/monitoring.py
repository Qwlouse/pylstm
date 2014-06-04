#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from pylstm.describable import Describable
from pylstm.error_functions import ClassificationError, LabelingError
from collections import OrderedDict


class Monitor(Describable):
    __undescribed__ = {'__name__'}  # the name is saved in the trainer

    def __init__(self, name=None, timescale='epoch', interval=1, verbose=None):
        self.timescale = timescale
        self.interval = interval
        if name is None:
            self.__name__ = self.__class__.__name__
        else:
            self.__name__ = name
        self.priority = 0
        self.verbose = verbose
        self.verbose_set = (verbose is not None)

    def start(self, net, stepper, verbose):
        if not self.verbose_set:
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
    __undescribed__ = {
        'weights': None
    }

    def __init__(self, error_log_name, filename=None, name=None, verbose=None):
        super(SaveBestWeights, self).__init__(name, 'epoch', 1, verbose)
        self.error_log_name = error_log_name
        self.filename = filename
        self.weights = None

    def __call__(self, epoch, net, stepper, logs):
        e = logs[self.error_log_name]
        min_error_idx = np.argmin(e)
        if min_error_idx == len(e) - 1:
            if self.filename is not None:
                if self.verbose:
                    print("Saving weights to {0}...".format(self.filename))
                np.save(self.filename, net.param_buffer)
            else:
                if self.verbose:
                    print("Caching weights")
                self.weights = net.param_buffer.copy()
        elif self.verbose:
            print("Last saved weigths after epoch {}".format(min_error_idx))

    def load_weights(self):
        return np.load(self.filename) if self.filename is not None \
            else self.weights


aggregate_error = lambda x: np.mean(x, axis=0)


def aggregate_class_error(x):
    e = np.sum(x, axis=1)
    return e[0] / e[1]


class MonitorError(Monitor):
    """
    Monitor the given error (aggregated over all sequences).
    """
    def __init__(self, data_iter, error_func=None, name=None,
                 aggregate=aggregate_error, timescale='epoch', interval=1):
        if name is None and error_func is not None:
            name = 'Monitor' + error_func.__name__
        super(MonitorError, self).__init__(name, timescale, interval)
        self.data_iter = data_iter
        self.error_func = error_func
        self.aggregate = aggregate

    def __call__(self, epoch, net, stepper, logs):
        error_func = self.error_func or net.error_func
        errors = []
        for x, t in self.data_iter(self.verbose):
            y = net.forward_pass(x)
            error, _ = error_func(y, t)
            errors.append(error)
        return self.aggregate(errors)


class MonitorClassificationError(MonitorError):
    def __init__(self, data_iter, name=None, timescale='epoch', interval=1):
        super(MonitorClassificationError, self).__init__(
            data_iter,
            error_func=ClassificationError,
            aggregate=aggregate_class_error,
            name=name, timescale=timescale, interval=interval)


class MonitorLabelingError(MonitorError):
    def __init__(self, data_iter, name=None, timescale='epoch', interval=1):
        super(MonitorLabelingError, self).__init__(
            data_iter,
            error_func=LabelingError,
            aggregate=aggregate_class_error,
            name=name, timescale=timescale, interval=interval)


class MonitorMultipleErrors(Monitor):
    """
    Monitor errors (aggregated over all sequences).
    """
    def __init__(self, data_iter, error_functions, name=None,
                 aggregators=aggregate_error, timescale='epoch', interval=1):
        super(MonitorMultipleErrors, self).__init__(name, timescale, interval)
        self.data_iter = data_iter
        self.error_functions = error_functions
        if isinstance(aggregators, (list, tuple)):
            self.aggregators = aggregators
        else:
            self.aggregators = [aggregators] * len(error_functions)

    def __call__(self, epoch, net, stepper, logs):
        errors = {e: [] for e in self.error_functions}
        for x, t in self.data_iter(self.verbose):
            y = net.forward_pass(x)
            for error_func in self.error_functions:
                error, _ = error_func(y, t)
                errors[error_func].append(error)

        return {err_f.__name__: agg(errors[err_f])
                for err_f, agg in zip(self.error_functions, self.aggregators)}


class MonitorPooledClassificationError(Monitor):
    """
        Monitor the classification error assuming one-hot encoding of targets
        with pooled targets with an odd pool size.
        """
    def __init__(self, data_iter, pool_size, name=None, timescale='epoch',
                 interval=1):
        super(MonitorPooledClassificationError, self).__init__(name, timescale,
                                                               interval)
        self.data_iter = data_iter
        self.pool_size = pool_size
    
    def __call__(self, epoch, net, stepper, logs):
        total_errors = 0
        total = 0
        for x, t in self.data_iter(self.verbose):
            relevant_from = (t.data.shape[2] / self.pool_size) * \
                            (self.pool_size // 2)
            relevant_to = relevant_from + (t.shape[2] / self.pool_size)
            t = t[:, :, relevant_from: relevant_to]
            y = net.forward_pass(x)
            y_win = y.argmax(2)
            t_win = t.argmax(2)  # fixme: offset by relevant_from?
            if t.mask is not None:
                total_errors += np.sum((y_win != t_win) * t.mask[:, :, 0])
                total += np.sum(t.mask)
            else:
                total_errors += np.sum((y_win != t_win))
                total += t.data.shape[0] * t.data.shape[1]
        error_fraction = total_errors / total
        return error_fraction


class PlotErrors(Monitor):
    """
    Open a window and plot the training and validation errors while training.
    """
    __undescribed__ = {'plt', 'fig', 'ax', 'lines'}

    def __init__(self, name=None, timescale='epoch', interval=1):
        super(PlotErrors, self).__init__(name, timescale, interval)
        import matplotlib.pyplot as plt
        self.plt = plt
        self.plt.ion()
        self.fig = None
        self.ax = None
        self.lines = None

    def start(self, net, stepper, verbose):
        super(PlotErrors, self).start(net, stepper, verbose)
        self.fig, self.ax = self.plt.subplots()
        self.ax.set_title('Training Progress')
        self.ax.set_xlabel('Epochs')
        self.ax.set_ylabel('Error')
        self.lines = dict()
        self.plt.show()

    def __call__(self, epoch, net, stepper, logs):
        for name, log in logs.items():
            if not isinstance(log, list) or not log:
                continue
            if name not in self.lines:
                line, = self.ax.plot(log, '-', label=name)
                self.lines[name] = line
            else:
                self.lines[name].set_ydata(log)
                self.lines[name].set_xdata(range(len(log)))

        self.ax.legend()
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()


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