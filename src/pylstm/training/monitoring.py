#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from pylstm.utils import ctc_best_path_decoding
from pylstm.describable import Describable
from .utils import levenshtein, get_min_err
from collections import OrderedDict


class Monitor(Describable):
    def __call__(self, epoch, net, stepper, training_errors, validation_errors):
        pass


class PrintError(Monitor):
    def __init__(self, timescale='epoch', interval=1):
        self.timescale = timescale
        self.interval = interval

    def __call__(self, epoch, training_errors, validation_errors, **_):
        if validation_errors:
            print("\nEpoch %d:\tTraining error = %0.4f Validation error = %0.4f"
                  % (epoch, training_errors[-1], validation_errors[-1]))
        else:
            print("\nEpoch %d:\tTraining error = %0.4f"
                  % (epoch, training_errors[-1]))


class SaveWeights(Monitor):
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


class SaveBestWeights(Monitor):
    """
    Check every epoch to see if the validation error (or training error if there
    is no validation error) is at it's minimum and if so, save the weights to
    the specified file.
    """
    __undescribed__ = {
        'weights': None
    }

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
        return np.load(self.filename) if self.filename is not None \
            else self.weights


class MonitorError(Monitor):
    """
    Monitor the given error (averaged over all sequences).
    """
    __undescribed__ = {
        'log': {'error': []}
    }

    def __init__(self, data_iter, error, name="", timescale='epoch',
                 interval=1):
        self.timescale = timescale
        self.interval = interval
        self.data_iter = data_iter
        self.error_func = error
        self.name = name
        self.log = {'error': []}

    def __call__(self, net, **_):
        errors = []
        for x, t in self.data_iter():
            y = net.forward_pass(x)
            error, _ = self.error_func(y, t)
            errors.append(error)

        mean_error = np.mean(errors)
        self.log['error'].append(mean_error)
        print(self.name, "= %0.4f" % mean_error)


class MonitorClassificationError(Monitor):
    """
    Monitor the classification error assuming one-hot encoding of targets.
    """
    __undescribed__ = {
        'log': {'classification_error': []}
    }

    def __init__(self, data_iter, name="", timescale='epoch', interval=1):
        self.timescale = timescale
        self.interval = interval
        self.data_iter = data_iter
        self.name = name
        self.log = {'classification_error': []}

    def __call__(self, net, **_):
        total_errors = 0
        total = 0
        for x, t in self.data_iter():
            assert t.targets_type in [('F', False), ('F', True),
                                      ('S', False), ('S', True)], \
                "Target type not suitable for classification error monitoring."
            y = net.forward_pass(x)
            y_win = y.argmax(2)
            if t.binarize_to is not None:
                t_win = t.data
            else:
                t_win = t.data.argmax(2)

            if t.mask is not None:
                total_errors += np.sum((y_win != t_win) * t.mask[:, :, 0])
                total += np.sum(t.mask)
            else:
                total_errors += np.sum((y_win != t_win))
                total += t.data.shape[0] * t.data.shape[1]
        error_fraction = total_errors / total
        self.log['classification_error'].append(error_fraction)
        print("{0:40}: Classification Error = {1:0.4f}\t ({2} / {3})".format(
            self.name, error_fraction, total_errors, total))


class MonitorPooledClassificationError(Monitor):
    """
        Monitor the classification error assuming one-hot encoding of targets
        with pooled targets with an odd pool size.
        """
    __undescribed__ = {
        'log': {'classification_error': []}
    }

    def __init__(self, data_iter, pool_size, name="", timescale='epoch',
                 interval=1):
        self.timescale = timescale
        self.interval = interval
        self.data_iter = data_iter
        self.pool_size = pool_size
        self.name = name
        self.log = {'classification_error': []}
    
    def __call__(self, net, **_):
        total_errors = 0
        total = 0
        for x, t in self.data_iter():
            relevant_from = (t.data.shape[2] / self.pool_size) * \
                            (self.pool_size // 2)
            relevant_to = relevant_from + (t.shape[2] / self.pool_size)
            t = t[:, :, relevant_from: relevant_to]
            y = net.forward_pass(x)
            y_win = y.argmax(2)
            t_win = t.argmax(2)
            if t.mask is not None:
                total_errors += np.sum((y_win != t_win) * t.mask[:, :, 0])
                total += np.sum(t.mask)
            else:
                total_errors += np.sum((y_win != t_win))
                total += t.data.shape[0] * t.data.shape[1]
        error_fraction = total_errors / total
        self.log['classification_error'].append(error_fraction)
        print(self.name, ":\tClassification Error = %0.4f\t (%d / %d)" %
              (error_fraction, total_errors, total))


class MonitorPhonemeError(Monitor):
    """
    Monitor the classification error assuming one-hot encoding of targets.
    """
    __undescribed__ = {'phoneme_error': []}

    def __init__(self, data_iter, name="", timescale='epoch', interval=1):
        self.timescale = timescale
        self.interval = interval
        self.data_iter = data_iter
        self.name = name
        self.log = {'phoneme_error': []}

    def __call__(self, net, **_):
        total_errors = 0
        total_length = 0
        for x, t in self.data_iter():
            y = net.forward_pass(x)
            lab = ctc_best_path_decoding(y)
            total_errors += levenshtein(lab, t.data[0])
            total_length += len(t.data[0])
        error_fraction = total_errors / total_length
        self.log['phoneme_error'].append(error_fraction)
        print(self.name, ':\tPhoneme Error = %0.4f\t (%d / %d)' %
                         (error_fraction, total_errors, total_length))


class PlotErrors(Monitor):
    """
    Open a window and plot the training and validation errors while training.
    """
    __undescribed__ = {'plt', 'fig', 'ax', 't_line', 'v_line', 'v_dot'}

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

    def __call__(self, training_errors, validation_errors, **_):
        if self.v_line is None and validation_errors:
                self.v_line, = self.ax.plot(validation_errors, 'b-',
                                            label='Validation Error')

        if self.t_line is None:
            if training_errors:
                self.t_line, = self.ax.plot(training_errors, 'g-',
                                            label='Training Error')
                if validation_errors:
                    min_ep, min_err = get_min_err(validation_errors)
                    self.v_dot, = self.ax.plot([min_ep], min_err, 'bo')
            self.ax.legend()
            self.fig.canvas.draw()
            return

        self.t_line.set_ydata(training_errors)
        self.t_line.set_xdata(range(len(training_errors)))
        if self.v_line is not None and validation_errors:
            self.v_line.set_ydata(validation_errors)
            self.v_line.set_xdata(range(len(validation_errors)))
            min_ep, min_err = get_min_err(validation_errors)
            self.v_dot.set_ydata([min_err])
            self.v_dot.set_xdata([min_ep])
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()


class PlotMonitors(Monitor):
    """
    Open a window and plot the log entries of the given monitors.
    """
    __undescribed__ = {
        'plt': None,
        'fig': None,
        'ax': None,
        'line_dict': dict()
    }

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


class MonitorLayerProperties(Monitor):
    """
    Monitor some properties of a layer.
    Also logs the initial values of the properties at creating time (before training).
    """
    __undescribed__ = {
        "log": OrderedDict()
    }

    def __init__(self, net, layer_name, display=True, timescale='epoch', interval=1):
        self.timescale = timescale
        self.interval = interval
        self.layer_name = layer_name
        self.display = display
        self.layer_type = net.layers[layer_name].get_typename()
        self.log = OrderedDict()

        for key, value in net.get_param_view_for(self.layer_name).items():

            # Log initial values
            self.log[self.layer_name + '_min_' + key] = [value.min()]
            self.log[self.layer_name + '_max_' + key] = [value.max()]
            if key.split('_')[-1] != 'bias':
                self.log[self.layer_name + '_min_sq_norm_' + key] = [((value**2).sum(1)).min()]
                self.log[self.layer_name + '_max_sq_norm_' + key] = [((value**2).sum(1)).max()]

    def __call__(self, net, **_):
        for key, value in net.get_param_view_for(self.layer_name).items():
            self.log[self.layer_name + '_min_' + key].append(value.min())
            self.log[self.layer_name + '_max_' + key].append(value.max())
            if key.split('_')[-1] != 'bias':
                self.log[self.layer_name + '_min_sq_norm_' + key].append(((value**2).sum(1)).min())
                self.log[self.layer_name + '_max_sq_norm_' + key].append(((value**2).sum(1)).max())

        if self.display:
            for key, value in self.log.items():
                print('{0:40}: {1}'.format(key, value[-1]))

