#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from copy import deepcopy
import wrapper as pw
import numpy as np


class Network(object):
    def __init__(self, layers, param_manager, fwd_state_manager, in_out_manager,
                 bwd_state_manager, delta_manager, error_func):
        self.layers = layers

        self.param_manager = param_manager
        self.grad_manager = deepcopy(param_manager)
        self.v_manager = deepcopy(param_manager)

        self.fwd_state_manager = fwd_state_manager
        self.r_fwd_state_manager = deepcopy(fwd_state_manager)
        self.bwd_state_manager = bwd_state_manager

        self.in_out_manager = in_out_manager
        self.r_in_out_manager = deepcopy(in_out_manager)
        self.delta_manager = delta_manager

        self.error_func = error_func

    @property
    def in_buffer(self):
        return self.in_out_manager.get_source_view("Input").as_array()

    @property
    def out_buffer(self):
        return self.in_out_manager.get_sink_view("Output").as_array()

    @property
    def param_buffer(self):
        return self.param_manager.buffer.as_array()

    @param_buffer.setter
    def param_buffer(self, buffer_view):
        """
        Set the parameter buffer that holds all the weights.
        """
        if isinstance(buffer_view, pw.Matrix):
            self.param_manager.initialize_buffer(buffer_view)
        else:
            self.param_manager.initialize_buffer(pw.Matrix(buffer_view))

    @property
    def grad_buffer(self):
        return self.grad_manager.buffer.as_array()

    def get_param_size(self):
        """
        Returns the total size of all parameters.
        """
        return self.param_manager.calculate_size()

    def get_input_size(self):
        return self.layers["Input"].get_output_size()

    def get_output_size(self):
        return self.layers["Output"].get_input_size()

    def get_param_view_for(self, name):
        return self.param_manager.get_source_view(name).as_array()

    def get_fwd_state_for(self, name):
        return self.fwd_state_manager.get_source_view(name)

    def get_bwd_state_for(self, name):
        return self.bwd_state_manager.get_source_view(name)

    def get_input_view_for(self, name):
        return self.in_out_manager.get_sink_view(name).as_array()

    def get_output_view_for(self, name):
        return self.in_out_manager.get_source_view(name).as_array()

    def get_in_deltas_view_for(self, name):
        return self.delta_manager.get_sink_view(name).as_array()

    def get_out_deltas_view_for(self, name):
        return self.delta_manager.get_source_view(name).as_array()

    def clear_internal_state(self):
        if self.fwd_state_manager.buffer:
            self.fwd_state_manager.buffer.as_array()[:] = 0.
        if self.bwd_state_manager.buffer:
            self.bwd_state_manager.buffer.as_array()[:] = 0.

    def __getitem__(self, item):
        """
        Get the layer with the given name.
        """
        return self.layers[item]

    def set_buffer_manager_dimensions(self, t, b):
        self.fwd_state_manager.set_dimensions(t, b)
        self.r_fwd_state_manager.set_dimensions(t, b)
        self.bwd_state_manager.set_dimensions(t, b)
        self.in_out_manager.set_dimensions(t, b)
        self.r_in_out_manager.set_dimensions(t, b)
        self.delta_manager.set_dimensions(t, b)

    def forward_pass(self, input_buffer):
        # determine dimensions and set buffer managers accordingly
        t, b, f = input_buffer.shape
        assert f == self.layers.values()[0].get_output_size()
        self.set_buffer_manager_dimensions(t, b)
        # inject the input buffer
        self.in_buffer[:] = input_buffer
        # execute all the intermediate layers
        for n, l in self.layers.items()[1:-1]:
            param = self.param_manager.get_source_view(n)
            fwd_state = self.fwd_state_manager.get_source_view(n)

            out = self.in_out_manager.get_source_view(n)
            input_view = self.in_out_manager.get_sink_view(n)

            l.forward(param, fwd_state, input_view, out)
        # read the output buffer
        return self.out_buffer

    def calculate_error(self, T):
        return self.error_func.evaluate(self.out_buffer, T)

    def backward_pass(self, T):
        delta_buffer = self.error_func.deriv(self.out_buffer, T)
        t, b, f = delta_buffer.shape
        # dims should already be set during forward_pass, but in any case...
        self.set_buffer_manager_dimensions(t, b)
        # clear all delta buffers
        self.delta_manager.clear_buffer()
        # inject delta_buffer
        out_view = self.delta_manager.get_sink_view("Output").as_array()
        out_view[:] = delta_buffer
        # execute all the intermediate layers backwards
        for n, l in self.layers.items()[-2:0:-1]:
            param = self.param_manager.get_source_view(n)
            fwd_state = self.fwd_state_manager.get_source_view(n)
            bwd_state = self.bwd_state_manager.get_source_view(n)

            out = self.in_out_manager.get_source_view(n)
            delta_in = self.delta_manager.get_sink_view(n)
            delta_out = self.delta_manager.get_source_view(n)

            l.backward(param, fwd_state, bwd_state, out, delta_in, delta_out)
        # read the final delta buffer
        return self.delta_manager.get_source_view("Input").as_array()

    def calc_gradient(self):
        self.grad_manager.initialize_buffer(
            pw.Matrix(self.get_param_size()))
        for n, l in self.layers.items()[-2:0:-1]:
            param = self.param_manager.get_source_view(n)
            grad = self.grad_manager.get_source_view(n)
            fwd_state = self.fwd_state_manager.get_source_view(n)
            bwd_state = self.bwd_state_manager.get_source_view(n)

            out = self.in_out_manager.get_source_view(n)
            input_view = self.in_out_manager.get_sink_view(n)
            delta_out = self.delta_manager.get_source_view(n)

            l.gradient(param, grad, fwd_state, bwd_state, out, input_view, delta_out)
        return self.grad_manager.buffer.as_array()

    def r_forward_pass(self, input_buffer, v_buffer):
        # determine dimensions and set buffer managers accordingly
        t, b, f = input_buffer.shape
        assert f == self.layers.values()[0].get_output_size()
        self.set_buffer_manager_dimensions(t, b)
        # inject the v value
        if isinstance(v_buffer, pw.Matrix):
            self.v_manager.initialize_buffer(v_buffer)
        else:
            self.v_manager.initialize_buffer(pw.Matrix(v_buffer))
        # inject the input buffer
        self.in_buffer[:] = input_buffer
        # set r input to 0
        self.r_in_out_manager.get_source_view("Input").as_array()[:] = 0.0
        # execute all the intermediate layers
        for n, l in self.layers.items()[1:-1]:
            param = self.param_manager.get_source_view(n)
            v = self.v_manager.get_source_view(n)
            fwd_state = self.fwd_state_manager.get_source_view(n)
            r_fwd_state = self.r_fwd_state_manager.get_source_view(n)

            out = self.in_out_manager.get_source_view(n)
            r_in = self.r_in_out_manager.get_sink_view(n)
            r_out = self.r_in_out_manager.get_source_view(n)
            input_view = self.in_out_manager.get_sink_view(n)

            l.Rpass(param, v, fwd_state, r_fwd_state, input_view, out, r_in, r_out)
            # read the output buffer
        return self.r_in_out_manager.get_sink_view("Output").as_array()

    def r_backward_pass(self, T, lambda_, mu):
        delta_buffer = self.r_in_out_manager.get_sink_view("Output").as_array()
        t, b, f = delta_buffer.shape
        # dims should already be set during forward_pass, but in any case...
        self.set_buffer_manager_dimensions(t, b)
        # inject delta_buffer
        out_view = self.delta_manager.get_sink_view("Output").as_array()
        out_view[:] = delta_buffer
        # execute all the intermediate layers backwards
        for n, l in self.layers.items()[-2:0:-1]:
            param = self.param_manager.get_source_view(n)
            fwd_state = self.fwd_state_manager.get_source_view(n)
            bwd_state = self.bwd_state_manager.get_source_view(n)

            delta_in = self.delta_manager.get_sink_view(n)
            delta_out = self.delta_manager.get_source_view(n)
            out = self.in_out_manager.get_source_view(n)
            l.backward(param, fwd_state, bwd_state, out, delta_in, delta_out)
            # l.Rbackward(param, fwd_state, bwd_state, delta_in, delta_out, r_fwd_state, lambda_, mu)
            # read the final delta buffer
        return self.delta_manager.get_source_view("Input").as_array()

    def hessian_pass(self, input_buffer, v_buffer, lambda_=0., mu=0.):
        t = input_buffer.shape[0]
        b = input_buffer.shape[1]
        T = np.zeros((t, b, self.get_output_size()))
        self.forward_pass(input_buffer)
        self.r_forward_pass(input_buffer, v_buffer)
        self.r_backward_pass(T, lambda_, mu)
        return self.calc_gradient()
