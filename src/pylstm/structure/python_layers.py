#!/usr/bin/env python
# coding=utf-8
"""
This File contains all layer types that are implemented in python.
See wrapper/py_layers.pyx for python wrappers around c++ layers.
"""
from __future__ import division, print_function, unicode_literals
import numpy as np

class LayerBase(object):
    """
    The base-class of all layer types defined in Python.
    """
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.skip_training = True

    def get_typename(self):
        return ""

    def get_input_buffer_size(self, time_length=1, batch_size=1):
        return self.in_size * time_length * batch_size

    def get_output_buffer_size(self, time_length=1, batch_size=1):
        return self.out_size * time_length * batch_size

    def get_parameter_size(self, time_length=1, batch_size=1):
        return 0

    def get_fwd_state_size(self, time_length, batch_size):
        return 0

    def get_bwd_state_size(self, time_length, batch_size1):
        return 0

    def create_input_view(self, input_buffer, time_length, batch_size):
        assert len(input_buffer) == self.get_input_buffer_size(time_length,
                                                               batch_size)
        return input_buffer.reshape(time_length, batch_size, self.in_size)

    def create_output_view(self, output_buffer, time_length, batch_size):
        assert len(output_buffer) == self.get_output_buffer_size(time_length,
                                                                 batch_size)
        return output_buffer.reshape(time_length, batch_size, self.out_size)

    def create_param_view(self, param_buffer, time_length=1, batch_size=1):
        return None

    def create_fwd_state(self, fwd_state_buffer, time_length, batch_size):
        return None

    def create_bwd_state(self, bwd_state_buffer, time_length, batch_size):
        return None

    def forward(self, param, fwd_state, in_view, out_view, training_pass):
        pass

    def backward(self, param, fwd_state, bwd_state, out_view, in_deltas,
                 out_deltas):
        pass

    def gradient(self, param, grad, fwd_state, bwd_state, out_view, in_view,
                 out_deltas):
        pass


class InputLayer(LayerBase):
    def get_typename(self):
        return 'InputLayer'


class NoOpLayer(LayerBase):
    """
    This is essentially a no-op layer.
    It just copies its input into its output.
    """
    def get_typename(self):
        return 'NoOpLayer'

    def forward(self, param, fwd_state, in_view, out_view, training_pass):
        out_view.as_array()[:] = in_view.as_array()

    def backward(self, param, fwd_state, bwd_state, out_view, in_deltas,
                 out_deltas):
        in_deltas.as_array()[:] = out_deltas.as_array()


class SquareLayer(LayerBase):
    """
    This layer squares every element.
    """
    def get_typename(self):
        return 'SquareLayer'

    def forward(self, param, fwd_state, in_view, out_view, training_pass):
        out_view.as_array()[:] = np.square(in_view.as_array())

    def backward(self, param, fwd_state, bwd_state, out_view, in_deltas,
                 out_deltas):
        in_deltas.as_array()[:] = 2*fwd_state.as_array()*out_deltas.as_array()


class DeltaInversionLayer(LayerBase):
    """
    A layer that does nothing during the forward pass, but reverses the sign
    of the deltas during the backward pass.
    """
    def get_typename(self):
        return 'DeltaInversionLayer'

    def forward(self, param, fwd_state, in_view, out_view, training_pass):
        out_view.as_array()[:] = in_view.as_array()

    def backward(self, param, fwd_state, bwd_state, out_view, in_deltas,
                 out_deltas):
        in_deltas.as_array()[:] = -out_deltas.as_array()
