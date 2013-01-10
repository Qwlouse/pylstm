#!/usr/bin/python
# coding=utf-8

import numpy as np
cimport numpy as np
cimport c_lstm_layer as clstm


cdef class LstmLayer:
    def __cinit__(self, int in_size, int out_size):
        self.in_size = in_size
        self.out_size = out_size

    def get_param_size(self):
        return clstm.LstmWeights(self.in_size, self.out_size).buffer_size()

    def get_output_size(self):
        return self.out_size

    def get_input_size(self):
        return self.in_size

    def get_internal_state_size(self, batch_size=1, time_length=1):
        b = clstm.LstmBuffers(self.in_size, self.out_size, batch_size, time_length)
        return b.buffer_size()

    def get_internal_error_state_size(self, batch_size=1, time_length=1):
        d = clstm.LstmDeltas(self.in_size, self.out_size, batch_size, time_length)
        return d.buffer_size()

    def create_input_view(self, input_buffer):
        return input_buffer

    def create_output_view(self, output_buffer):
        return output_buffer

    def create_param_view(self, param_buffer):
        params = clstm.LstmWeights(self.in_size, self.out_size)
        params.allocate(param_buffer)
        return params

    def create_internal_view(self, internal_buffer):
        batch_size = internal_buffer.batch_size()
        time_length = internal_buffer.time_length()
        internal = clstm.LstmBuffers(self.in_size, self.out_size, batch_size, time_length)
        internal.allocate(internal_buffer)
        return internal

    def create_internal_error_view(self, internal_error_buffer):
        batch_size = internal_error_buffer.batch_size()
        time_length = internal_error_buffer.time_length()
        deltas = clstm.LstmDeltas(self.in_size, self.out_size, batch_size, time_length)
        deltas.allocate(internal_error_buffer)
        return deltas

    def forward(self, input, param, internal, output):
        clstm.lstm_forward(param, internal, input, output)
