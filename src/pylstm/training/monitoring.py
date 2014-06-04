#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from pylstm.utils import ctc_best_path_decoding
from pylstm.describable import Describable
from .utils import levenshtein
from collections import OrderedDict


class Monitor(Describable):
    __undescribed__ = {'__name__'}  # the name is saved in the trainer

    def __init__(self, name=None, timescale='epoch', interval=1):
        self.timescale = timescale
        self.interval = interval
        if name is None:
            self.__name__ = self.__class__.__name__
        else:
            self.__name__ = name
        self.priority = 0

    def start(self, net, stepper):
        pass

    def __call__(self, epoch, net, stepper, logs):
        pass


class PrintError(Monitor):
    def __call__(self, epoch, net, stepper, logs):
        print("\nEpoch %d:\tTraining error = %0.4f"
              % (epoch, logs['training_errors'][-1]))


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

    def __init__(self, error_log_name, filename=None, name=None, verbose=True):
        super(SaveBestWeights, self).__init__(name, 'epoch', 1)
        self.error_log_name = error_log_name
        self.filename = filename
        self.weights = None
        self.verbose = verbose

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


class MonitorError(Monitor):
    """
    Monitor the given error (averaged over all sequences).
    """
    def __init__(self, data_iter, error_func=None, name=None,
                 timescale='epoch', interval=1):
        if name is None and error_func is not None:
            name = 'Monitor' + error_func.__name__
        super(MonitorError, self).__init__(name, timescale, interval)
        self.data_iter = data_iter
        self.error_func = error_func

    def __call__(self, epoch, net, stepper, logs):
        error_func = self.error_func or net.error_func
        errors = []
        for x, t in self.data_iter():
            y = net.forward_pass(x)
            error, _ = error_func(y, t)
            errors.append(error)

        mean_error = np.mean(errors)
        print(self.__name__, "= %0.4f" % mean_error)  # todo: print in trainer?
        return mean_error


class MonitorLabelError(Monitor):
    """
    Monitor the label-error using ctc_best_path_decoding and
    levenshtein distance.
    """
    def __init__(self, data_iter, name=None, timescale='epoch', interval=1):
        super(MonitorLabelError, self).__init__(name, timescale, interval)
        self.data_iter = data_iter

    def __call__(self, epoch, net, stepper, logs):
        total_errors = 0
        total_length = 0
        for x, t in self.data_iter():
            assert t.targets_type == ('L', True)
            y = net.forward_pass(x)
            lab = ctc_best_path_decoding(y)
            total_errors += levenshtein(lab, t.data[0])
            total_length += len(t.data[0])
        error_fraction = total_errors / total_length
        print(self.__name__, ':\tLabel Error = %0.4f\t (%d / %d)' %
              (error_fraction, total_errors, total_length))
        return error_fraction


class MonitorClassificationError(Monitor):
    """
    Monitor the classification error assuming one-hot encoding of targets.
    """
    def __init__(self, data_iter, name=None, timescale='epoch', interval=1):
        super(MonitorClassificationError, self).__init__(name, timescale,
                                                         interval)
        self.data_iter = data_iter

    def __call__(self, epoch, net, stepper, logs):
        total_errors = 0
        total = 0
        for x, t in self.data_iter():
            assert t.targets_type == ('F', True), \
                "Target type not suitable for classification error monitoring."
            y = net.forward_pass(x)
            y_win = y.argmax(2)
            t_win = t.data[:, :, 0]
            if t.mask is not None:
                total_errors += np.sum((y_win != t_win) * t.mask[:, :, 0])
                total += np.sum(t.mask)
            else:
                total_errors += np.sum((y_win != t_win))
                total += t.data.shape[0] * t.data.shape[1]
        error_fraction = total_errors / total
        print("{0:40}: Classification Error = {1:0.4f}\t ({2} / {3})".format(
            self.__name__, error_fraction, total_errors, total))
        return error_fraction


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
        for x, t in self.data_iter():
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
        print(self.__name__, ":\tClassification Error = %0.4f\t (%d / %d)" %
              (error_fraction, total_errors, total))
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

    def start(self, net, stepper):
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
    def __init__(self, layer_name, verbose=True, timescale='epoch',
                 interval=1, name=None):
        if name is None:
            name = "Monitor{}Properties".format(layer_name)
        super(MonitorLayerProperties, self).__init__(name, timescale, interval)
        self.layer_name = layer_name
        self.verbose = verbose

    def __call__(self, epoch, net, stepper, logs):
        log = OrderedDict()
        for key, value in net.get_param_view_for(self.layer_name).items():
            log['min_' + key] = value.min()
            log['max_' + key] = value.max()
            #if key.split('_')[-1] != 'bias':
            if value.shape[1] > 1:
                log['min_sq_norm_' + key] = (value ** 2).sum(1).min()
                log['max_sq_norm_' + key] = (value ** 2).sum(1).max()

        if self.verbose:
            print(self.layer_name)
            for key, value in log.items():
                print('  {0:38}: {1}'.format(key, value))

        return log