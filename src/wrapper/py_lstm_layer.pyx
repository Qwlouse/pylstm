#!/usr/bin/python
# coding=utf-8

import numpy as np
cimport numpy as np
cimport c_lstm_layer as clstm
cimport c_matrix as cm
cimport py_matrix as pm


cdef class LstmParamBuffer:
    def __cinit__(self, int in_size, int out_size):
        self.thisptr = new clstm.LstmWeights(in_size, out_size)

cdef class LstmInternalBuffer:
    def __cinit__(self, int in_size, int out_size, int batch_size, int time_length):
        self.thisptr = new clstm.LstmBuffers(in_size, out_size, batch_size, time_length)

cdef class LstmErrorBuffer:
    def __cinit__(self, int in_size, int out_size, int batch_size, int time_length):
        self.thisptr = new clstm.LstmDeltas(in_size, out_size, batch_size, time_length)


cdef class LstmLayer:
    def __cinit__(self, int in_size, int out_size):
        self.in_size = in_size
        self.out_size = out_size

    def get_param_size(self):
        return self.get_weight_size()

    def get_output_size(self):
        return self.out_size

    def get_input_size(self):
        return self.in_size

#    def get_internal_state_size(self, batch_size=1, time_length=1):
#        return clstm.LstmBuffers(self.in_size, self.out_size, batch_size,
#            time_length).buffer_size()
#
#    def get_internal_error_state_size(self, batch_size=1, time_length=1):
#        return clstm.LstmDeltas(self.in_size, self.out_size, batch_size,
#            time_length).buffer_size()

    def create_input_view(self, input_buffer):
        return input_buffer

    def create_output_view(self, output_buffer):
        return output_buffer

    def create_param_view(self, pm.MatrixCPU param_buffer):
        params = LstmParamBuffer(self.in_size, self.out_size)
        #params.thisptr.allocate(param_buffer.get_2d_view())
        return params

    def create_internal_view(self, pm.MatrixCPU internal_buffer):
        batch_size = internal_buffer.batch_size()
        time_length = internal_buffer.time_length()
        internal = LstmInternalBuffer(self.in_size, self.out_size, batch_size, time_length)
        #internal.thisptr.allocate(internal_buffer.get_2d_view())
        return internal

    def create_internal_error_view(self, pm.MatrixCPU internal_error_buffer):
        batch_size = internal_error_buffer.batch_size()
        time_length = internal_error_buffer.time_length()
        deltas = LstmErrorBuffer(self.in_size, self.out_size, batch_size, time_length)
        #deltas.thisptr.allocate(internal_error_buffer.get_2d_view())
        return deltas

    def forward(self, pm.MatrixCPU input, LstmParamBuffer param, LstmInternalBuffer internal, pm.MatrixCPU output):
        lstm_forward(param, internal, input, output)
