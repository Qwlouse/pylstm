#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from copy import deepcopy
import wrapper as pw
import numpy as np


class Network(object):
    def __init__(self, layers, weight_manager, intern_manager, in_out_manager,
                 intern_delta_manager, delta_manager, error_func):
        self.layers = layers

        self.weight_manager = weight_manager
        self.grad_manager = deepcopy(weight_manager)
        self.v_manager = deepcopy(weight_manager)

        self.intern_manager = intern_manager
        self.r_intern_manager = deepcopy(intern_manager)
        self.intern_delta_manager = intern_delta_manager

        self.in_out_manager = in_out_manager
        self.r_in_out_manager = deepcopy(in_out_manager)
        self.delta_manager = delta_manager

        self.error_func = error_func

    def get_param_size(self):
        """
        Returns the total size of all parameters.
        """
        return self.weight_manager.calculate_size()

    def get_input_size(self):
        return self.layers["Input"].get_output_size()

    def get_output_size(self):
        return self.layers["Output"].get_input_size()

    def get_param_view_for(self, name):
        return self.weight_manager.get_source_view(name)

    def get_intern_view_for(self, name):
        return self.intern_manager.get_source_view(name)

    def get_output_view_for(self, name):
        return self.in_out_manager.get_source_view(name)

    def clear_internal_state(self):
        if self.intern_manager.buffer:
            self.intern_manager.buffer.as_array()[:] = 0.
        if self.intern_delta_manager.buffer:
            self.intern_delta_manager.buffer.as_array()[:] = 0.

    def __getitem__(self, item):
        """
        Get the layer with the given name.
        """
        return self.layers[item]

    def set_param_buffer(self, buffer_view):
        """
        Set the parameter buffer that holds all the weights.
        """
        if isinstance(buffer_view, pw.Buffer):
            self.weight_manager.initialize_buffer(buffer_view)
        else:
            self.weight_manager.initialize_buffer(pw.Buffer(buffer_view))

    def get_param_buffer(self):
        return self.weight_manager.buffer

    def set_buffer_manager_dimensions(self, t, b):
        self.intern_manager.set_dimensions(t, b)
        self.r_intern_manager.set_dimensions(t, b)
        self.intern_delta_manager.set_dimensions(t, b)
        self.in_out_manager.set_dimensions(t, b)
        self.r_in_out_manager.set_dimensions(t, b)
        self.delta_manager.set_dimensions(t, b)

    def forward_pass(self, input_buffer):
        # determine dimensions and set buffer managers accordingly
        t, b, f = input_buffer.shape
        assert f == self.layers.values()[0].get_output_size()
        self.set_buffer_manager_dimensions(t, b)
        # inject the input buffer
        self.in_out_manager.get_source_view("Input").as_array()[:] = input_buffer # TODO factor this out as self.in_buffer property
        # execute all the intermediate layers
        for n, l in self.layers.items()[1:-1]:
            param = self.weight_manager.get_source_view(n)
            internal = self.intern_manager.get_source_view(n)

            out = self.in_out_manager.get_source_view(n)
            input_view = self.in_out_manager.get_sink_view(n)

            l.forward(param, internal, input_view, out)
        # read the output buffer
        return self.in_out_manager.get_sink_view("Output") # TODO factor this out as self.out_buffer property

    def calculate_error(self, T):
        X = self.in_out_manager.get_sink_view("Output")
        return self.error_func.evaluate(X, T)

    def backward_pass(self, T):
        X = self.in_out_manager.get_sink_view("Output")
        delta_buffer = self.error_func.deriv(X, T)
        t, b, f = delta_buffer.shape
        # dims should already be set during forward_pass, but in any case...
        self.set_buffer_manager_dimensions(t, b)
        # inject delta_buffer
        out_view = self.delta_manager.get_sink_view("Output").as_array()
        out_view[:] = delta_buffer
        # execute all the intermediate layers backwards
        for n, l in self.layers.items()[-2:0:-1]:
            param = self.weight_manager.get_source_view(n)
            internal = self.intern_manager.get_source_view(n)
            intern_delta = self.intern_delta_manager.get_source_view(n)

            out = self.in_out_manager.get_source_view(n)
            delta_in = self.delta_manager.get_sink_view(n)
            delta_out = self.delta_manager.get_source_view(n)

            l.backward(param, internal, intern_delta, out, delta_in, delta_out)
        # read the final delta buffer
        return self.delta_manager.get_source_view("Input")

    def calc_gradient(self):
        self.grad_manager.initialize_buffer(
            pw.Buffer(self.get_param_size()))
        for n, l in self.layers.items()[-2:0:-1]:
            param = self.weight_manager.get_source_view(n)
            grad = self.grad_manager.get_source_view(n)
            internal = self.intern_manager.get_source_view(n)
            intern_delta = self.intern_delta_manager.get_source_view(n)

            out = self.in_out_manager.get_source_view(n)
            input_view = self.in_out_manager.get_sink_view(n)
            delta_out = self.delta_manager.get_source_view(n)

            l.gradient(param, grad, internal, intern_delta, out, input_view, delta_out)
        return self.grad_manager.buffer

    def r_forward_pass(self, input_buffer, v_buffer):
        # determine dimensions and set buffer managers accordingly
        t, b, f = input_buffer.shape
        assert f == self.layers.values()[0].get_output_size()
        self.set_buffer_manager_dimensions(t, b)
        # inject the input buffer
        self.in_out_manager.get_source_view("Input").as_array()[:] = input_buffer
        # execute all the intermediate layers
        for n, l in self.layers.items()[1:-1]:
            param = self.weight_manager.get_source_view(n)
            v = self.v_manager.get_source_view(n)
            internal = self.intern_manager.get_source_view(n)
            r_internal = self.r_intern_manager.get_source_view(n)

            out = self.in_out_manager.get_source_view(n)
            r_out = self.r_in_out_manager.get_source_view(n)
            input_view = self.in_out_manager.get_sink_view(n)

            l.Rpass(param, v, internal, r_internal, input_view, out, r_out)
            # read the output buffer
        return self.r_in_out_manager.get_sink_view("Output")

    def r_backward_pass(self, T, lambda_, mu):
        X = self.in_out_manager.get_sink_view("Output")
        delta_buffer = self.error_func.deriv(X, T)
        t, b, f = delta_buffer.shape
        # dims should already be set during forward_pass, but in any case...
        self.set_buffer_manager_dimensions(t, b)
        # inject delta_buffer
        out_view = self.delta_manager.get_sink_view("Output").as_array()
        out_view[:] = delta_buffer
        # execute all the intermediate layers backwards
        for n, l in self.layers.items()[-2:0:-1]:
            param = self.weight_manager.get_source_view(n)
            internal = self.intern_manager.get_source_view(n)
            r_internal = self.r_intern_manager.get_source_view(n)
            intern_delta = self.intern_delta_manager.get_source_view(n)

            delta_in = self.delta_manager.get_sink_view(n)
            delta_out = self.delta_manager.get_source_view(n)

            l.Rbackward(param, internal, intern_delta, delta_in, delta_out, r_internal, lambda_, mu)
            # read the final delta buffer
        return self.delta_manager.get_source_view("Input")

    def hessian_pass(self, input_buffer, v_buffer, lambda_=0., mu=0.):
        t = input_buffer.shape[0]
        b = input_buffer.shape[1]
        T = np.zeros((t, b, self.get_output_size()))
        self.forward_pass(input_buffer)
        self.r_forward_pass(input_buffer, v_buffer)
        self.r_backward_pass(T, lambda_, mu)


