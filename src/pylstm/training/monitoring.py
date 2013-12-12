#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np


def MonitorFunction(timescale='epoch', interval=1):
    """
    Decorator to adjust the interval and timescale for a monitoring function.
    Example:

    @MonitorFunction('update', 5)
    def foo(*args, **kwargs):
        print('bar')
    """
    def decorator(f):
        f.timescale = timescale
        f.interval = interval
        return f
    return decorator


def print_error_per_epoch(epoch, training_errors, validation_errors, **_):
    if len(validation_errors) == 0:
        print("\nEpoch %d:\tTraining error = %0.4f" % (epoch,
                                                       training_errors[-1]))
    else:
        print("\nEpoch %d:\tTraining error = %0.4f Validation error = %0.4f" %
              (epoch, training_errors[-1], validation_errors[-1]))


class SaveWeights(object):
    """
    Save the weights of the network to the given file on every call.
    Default is to save them once per epoch, but this can be configured using
    the timescale and interval parameters.
    """
    def __init__(self, filename, timescale='epoch', interval=1):
        self.timescale = timescale
        self.interval = interval
        self.filename = filename

    def __call__(self, net, **_):
        np.save(self.filename, net.param_buffer)

    def load_weights(self):
        return np.load(self.filename)


class SaveBestWeights(object):
    """
    Check every epoch to see if the validation error (or training error if there
    is no validation error) is at it's minimum and if so, save the weights to
    the specified file.
    """
    def __init__(self, filename=None, verbose=True):
        self.timescale = 'epoch'
        self.interval = 1
        self.filename = filename
        self.weights = None
        self.verbose = verbose

    def __call__(self, net, training_errors, validation_errors, **_):
        e = validation_errors if len(validation_errors) > 0 else training_errors
        if np.argmin(e) == len(e) - 1:
            if self.filename is not None:
                if self.verbose:
                    print("Saving weights to {0}...".format(self.filename))
                np.save(self.filename, net.param_buffer)
            else:
                if self.verbose:
                    print("Caching weights")
                self.weights = net.param_buffer.copy()

    def load_weights(self):
        return np.load(self.filename) if self.filename is not None else self.weights


class MonitorError(object):
    """
    Monitor the given error.
    """
    def __init__(self, data_iter, error, name="", timescale='epoch', interval=1):
        self.timescale = timescale
        self.interval = interval
        self.data_iter = data_iter
        self.error_func = error
        self.name = name
        self.log = dict()
        self.log['error'] = []

    def __call__(self, net, **_):
        errors = []
        for x, t, m in self.data_iter():
            y = net.forward_pass(x)
            error, _ = self.error_func(y, t, m)
            errors.append(error)

        mean_error = np.mean(errors)
        self.log['error'].append(mean_error)
        print(self.name, "= %0.4f" % mean_error)


class MonitorClassificationError(object):
    """
    Monitor the classification error assuming one-hot encoding of targets.
    """
    def __init__(self, data_iter, name="", timescale='epoch', interval=1):
        self.timescale = timescale
        self.interval = interval
        self.data_iter = data_iter
        self.name = name
        self.log = dict()
        self.log['classification_error'] = []

    def __call__(self, net, **_):
        total_errors = 0
        total = 0
        for x, t, m in self.data_iter():
            y = net.forward_pass(x)
            y_win = y.argmax(2)
            if t.binarize_to is not None:
                t_win = t.data
            else:
                t_win = t.data.argmax(2)

            if m is not None:
                total_errors += np.sum((y_win != t_win) * m[:, :, 0])
                total += np.sum(m)
            else:
                total_errors += np.sum((y_win != t_win))
                total += t.shape[0] * t.shape[1]
        error_fraction = total_errors / total
        self.log['classification_error'].append(error_fraction)
        print(self.name, ":\tClassification Error = %0.4f\t (%d / %d)" %
                         (error_fraction, total_errors, total))


class MonitorPooledClassificationError(object):
    """
        Monitor the classification error assuming one-hot encoding of targets
        with pooled targets with an odd pool size.
        """
    def __init__(self, data_iter, pool_size, name="", timescale='epoch', interval=1):
        self.timescale = timescale
        self.interval = interval
        self.data_iter = data_iter
        self.pool_size = pool_size
        self.name = name
        self.log = dict()
        self.log['classification_error'] = []
    
    def __call__(self, net, **_):
        total_errors = 0
        total = 0
        for x, t, m in self.data_iter():
            relevant_from = (t.shape[2] / self.pool_size) * (self.pool_size // 2)
            relevant_to = relevant_from + (t.shape[2] / self.pool_size)
            t = t[:, :, relevant_from: relevant_to]
            y = net.forward_pass(x)
            y_win = y.argmax(2)
            t_win = t.argmax(2)
            if m is not None:
                total_errors += np.sum((y_win != t_win) * m[:, :, 0])
                total += np.sum(m)
            else:
                total_errors += np.sum((y_win != t_win))
                total += t.shape[0] * t.shape[1]
        error_fraction = total_errors / total
        self.log['classification_error'].append(error_fraction)
        print(self.name, ":\tClassification Error = %0.4f\t (%d / %d)" %
              (error_fraction, total_errors, total))


class MonitorPhonemeError(object):
    """
    Monitor the classification error assuming one-hot encoding of targets.
    """
    def __init__(self, data_iter, name="", timescale='epoch', interval=1):
        self.timescale = timescale
        self.interval = interval
        self.data_iter = data_iter
        self.name = name
        self.log = dict()
        self.log['phoneme_error'] = []

    def __call__(self, net, **_):
        total_errors = 0
        total_length = 0
        for x, t, m in self.data_iter():
            y = net.forward_pass(x)
            lab = ctc_best_path_decoding(y)
            total_errors += levenshtein(lab, t.data[0])
            total_length += len(t.data[0])
        error_fraction = total_errors / total_length
        self.log['phoneme_error'].append(error_fraction)
        print(self.name, ':\tPhoneme Error = %0.4f\t (%d / %d)' %
                         (error_fraction, total_errors, total_length))


def ctc_best_path_decoding(Y):
    assert Y.shape[1] == 1
    Y_win = Y.argmax(2).reshape(Y.shape[0])
    t = []
    blank = True
    for y in Y_win:
        if blank is True and y != 0:
            t.append(y - 1)
            blank = False
        elif blank is False:
            if y == 0:
                blank = True
            elif y - 1 != t[-1]:
                t.append(y - 1)
    return t


def levenshtein(seq1, seq2):
    oneago = None
    thisrow = range(1, len(seq2) + 1) + [0]
    for x in xrange(len(seq1)):
        twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
        for y in xrange(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
    return thisrow[len(seq2) - 1]


class PlotErrors(object):
    """
    Open a window and plot the training and validation errors while training.
    """
    def __init__(self, timescale='epoch', interval=1):
        self.timescale = timescale
        self.interval = interval
        import matplotlib.pyplot as plt
        self.plt = plt
        self.plt.ion()
        self.fig, self.ax = self.plt.subplots()
        self.ax.set_title('Training Progress')
        self.ax.set_xlabel('Epochs')
        self.ax.set_ylabel('Error')
        self.t_line = None
        self.v_line = None
        self.v_dot = None
        self.plt.show()

    def _get_min_err(self, errors):
        min_epoch = np.argmin(errors)
        return min_epoch, errors[min_epoch]

    def __call__(self, training_errors, validation_errors, **_):
        if self.v_line is None and validation_errors:
                self.v_line, = self.ax.plot(validation_errors, 'b-',
                                            label='Validation Error')

        if self.t_line is None:
            if training_errors:
                self.t_line, = self.ax.plot(training_errors, 'g-',
                                            label='Training Error')
                min_ep, min_err = self._get_min_err(validation_errors)
                self.v_dot, = self.ax.plot([min_ep], min_err, 'bo')
            self.ax.legend()
            self.fig.canvas.draw()
            return

        self.t_line.set_ydata(training_errors)
        self.t_line.set_xdata(range(len(training_errors)))
        if self.v_line is not None:
            self.v_line.set_ydata(validation_errors)
            self.v_line.set_xdata(range(len(validation_errors)))
            min_ep, min_err = self._get_min_err(validation_errors)
            self.v_dot.set_ydata([min_err])
            self.v_dot.set_xdata([min_ep])
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()


class PlotMonitors(object):
    """
    Open a window and plot the log entries of the given monitors.
    """
    def __init__(self, monitors, timescale='epoch', interval=1):
        self.timescale = timescale
        self.interval = interval
        self.monitors = monitors
        import matplotlib.pyplot as plt
        self.plt = plt
        self.plt.ion()
        self.fig, self.ax = self.plt.subplots()
        self.ax.set_title('Plotting Monitors')
        self.ax.set_xlabel('Epochs')
        self.ax.set_ylabel('Error')
        self.line_dict = dict()
        self.plt.show()

    def __call__(self, **_):

        for m in self.monitors:
            for k in m.log:
                n = m.name + '.' + k
                if n not in self.line_dict and m.log[k]:
                        self.line_dict[n], = self.ax.plot(m.log[k], '-', label=n)

            self.ax.legend()
            self.fig.canvas.draw()

        for m in self.monitors:
            for k in m.log:
                n = m.name + '.' + k
                self.line_dict[n].set_ydata(m.log[k])
                self.line_dict[n].set_xdata(range(len(m.log[k])))

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
