#!/usr/bin/python
# coding=utf-8
import numpy as np
cimport numpy as np
cimport c_lstm_layer as clstm
cimport c_matrix as cm
from cython.operator cimport dereference as deref

cdef class LstmParamBuffer:
    cdef clstm.LstmWeights* thisptr
    def __cinit__(self, int in_size, int out_size):
        self.thisptr = new clstm.LstmWeights(in_size, out_size)

    def __dealloc__(self):
        del self.thisptr

cdef class LstmInternalBuffer:
    cdef clstm.LstmBuffers* thisptr
    def __cinit__(self, int in_size, int out_size, int batch_size, int time_length):
        self.thisptr = new clstm.LstmBuffers(in_size, out_size, batch_size, time_length)

    def __dealloc__(self):
        del self.thisptr

cdef class LstmErrorBuffer:
    cdef clstm.LstmDeltas* thisptr
    def __cinit__(self, int in_size, int out_size, int batch_size, int time_length):
        self.thisptr = new clstm.LstmDeltas(in_size, out_size, batch_size, time_length)

    def __dealloc__(self):
        del self.thisptr

cdef class LstmLayer:
    cdef int in_size
    cdef int out_size

    def __cinit__(self, int in_size, int out_size):
        self.in_size = in_size
        self.out_size = out_size

    def get_output_size(self, time_length=1, batch_size=1):
        return self.out_size

    def get_input_size(self):
        return self.in_size

    def get_param_size(self, time_length=1, batch_size=1):
        return clstm.LstmWeights(self.in_size, self.out_size).buffer_size()

    def get_internal_state_size(self, time_length=1, batch_size=1):
        return clstm.LstmBuffers(self.in_size, self.out_size, batch_size, time_length).buffer_size()

    def get_internal_error_state_size(self, time_length=1, batch_size=1):
        return clstm.LstmDeltas(self.in_size, self.out_size, batch_size, time_length).buffer_size()

    def create_input_view(self, input_buffer, time_length=1, batch_size=1):
        return input_buffer

    def create_output_view(self, output_buffer, time_length=1, batch_size=1):
        return output_buffer

    def create_param_view(self, BufferView param_buffer, time_length=1, batch_size=1):
        params = LstmParamBuffer(self.in_size, self.out_size)
        params.thisptr.allocate(param_buffer.flatten2D())
        return params

    def create_internal_view(self, BufferView internal_buffer, time_length=1, batch_size=1):
        internal = LstmInternalBuffer(self.in_size, self.out_size, batch_size, time_length)
        internal.thisptr.allocate(internal_buffer.flatten2D())
        return internal

    def create_internal_error_view(self, BufferView internal_error_buffer, time_length=1, batch_size=1):
        deltas = LstmErrorBuffer(self.in_size, self.out_size, batch_size, time_length)
        deltas.thisptr.allocate(internal_error_buffer.flatten2D())
        return deltas

    def forward(self, LstmParamBuffer param, LstmInternalBuffer internal, BufferView input, BufferView output):
        clstm.lstm_forward(deref(param.thisptr), deref(internal.thisptr), input.view, output.view)

    def backward(self, LstmParamBuffer param, LstmInternalBuffer internal, LstmErrorBuffer err, BufferView output, BufferView in_deltas, BufferView out_deltas):
        clstm.lstm_backward(deref(param.thisptr), deref(internal.thisptr), deref(err.thisptr), output.view, in_deltas.view, out_deltas.view)

