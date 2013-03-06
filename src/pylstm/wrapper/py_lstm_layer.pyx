#!/usr/bin/python
# coding=utf-8
cimport c_lstm_layer as clstm
cimport c_matrix as cm
from cython.operator cimport dereference as deref
from py_matrix cimport Buffer

cdef class BufferContainer:
    cdef clstm.ViewContainer* this_ptr 
    
    def __cinit__(self):
        self.this_ptr = NULL
        
    def __dealloc__(self):
        del self.this_ptr

cdef create_BufferContainer(clstm.ViewContainer* c):
    bc = BufferContainer()
    bc.this_ptr = c
    return bc


cdef class BaseLayer:
    cdef int in_size
    cdef int out_size
    cdef clstm.BaseLayer* layer

    def __cinit__(self):
        self.layer = NULL
    
    def __dealloc(self):
        del self.layer

    def get_output_size(self):
        return self.layer.out_size

    def get_input_size(self):
        return self.layer.in_size

    def get_input_buffer_size(self, time_length=1, batch_size=1):
        return self.in_size * time_length * batch_size

    def get_output_buffer_size(self, time_length=1, batch_size=1):
        return self.out_size * time_length * batch_size

    def get_param_size(self, time_length=1, batch_size=1):
        return self.layer.get_weight_size()

    def get_internal_state_size(self, time_length=1, batch_size=1):
        return self.layer.get_fwd_state_size(batch_size, time_length)

    def get_internal_error_state_size(self, time_length=1, batch_size=1):
        return self.layer.get_bwd_state_size(batch_size, time_length)

    def create_input_view(self, input_buffer, time_length=1, batch_size=1):
        assert len(input_buffer) == self.get_input_buffer_size(time_length, batch_size)
        return input_buffer.reshape(time_length, batch_size, self.in_size)

    def create_output_view(self, output_buffer, time_length=1, batch_size=1):
        assert len(output_buffer) == self.get_output_buffer_size(time_length, batch_size)
        return output_buffer.reshape(time_length, batch_size, self.out_size)

    def create_param_view(self, Buffer param_buffer, time_length=1, batch_size=1):
        cdef clstm.ViewContainer* params = self.layer.create_weights_view(param_buffer.view)
        return create_BufferContainer(params)
        
    def create_internal_view(self, Buffer internal_buffer, time_length=1, batch_size=1):
        cdef clstm.ViewContainer* internal = self.layer.create_fwd_state_view(internal_buffer.view, batch_size, time_length)
        return create_BufferContainer(internal)

    def create_internal_error_view(self, Buffer internal_error_buffer, time_length=1, batch_size=1):
        cdef clstm.ViewContainer* deltas = self.layer.create_bwd_state_view(internal_error_buffer.view, batch_size, time_length)
        return create_BufferContainer(deltas)
"""
    def forward(self, LstmParamBuffer param, LstmInternalBuffer internal, BufferView input, BufferView output):
        clstm.lstm_forward(deref(param.thisptr), deref(internal.thisptr), input.view, output.view)

    def backward(self, LstmParamBuffer param, LstmInternalBuffer internal, LstmErrorBuffer err, BufferView output, BufferView in_deltas, BufferView out_deltas):
        clstm.lstm_backward(deref(param.thisptr), deref(internal.thisptr), deref(err.thisptr), output.view, in_deltas.view, out_deltas.view)

    def gradient(self, LstmParamBuffer param, LstmParamBuffer grad, LstmInternalBuffer internal, LstmErrorBuffer err, BufferView output, BufferView input):
        clstm.lstm_grad(deref(param.thisptr), deref(grad.thisptr), deref(internal.thisptr), deref(err.thisptr), output.view, input.view)
"""

def create_layer(name, in_size, out_size):
    l = BaseLayer()
    l.layer = clstm.create_layer(name, in_size, out_size)
    return l
