# coding=utf-8
cimport c_lstm_layer as clstm
cimport c_matrix as cm
from cython.operator cimport dereference as deref
from py_matrix cimport BufferView

cdef class FwdParamBuffer(object):
    cdef clstm.FwdWeights* thisptr
    def __cinit__(self, int in_size, int out_size):
        self.thisptr = new clstm.FwdWeights(in_size, out_size)

    def __dealloc__(self):
        del self.thisptr

cdef class FwdInternalBuffer(object):
    cdef clstm.FwdBuffers* thisptr
    def __cinit__(self, int in_size, int out_size, int batch_size, int time_length):
        self.thisptr = new clstm.FwdBuffers(in_size, out_size, batch_size, time_length)

    def __dealloc__(self):
        del self.thisptr

cdef class FwdErrorBuffer(object):
    cdef clstm.FwdDeltas* thisptr
    def __cinit__(self, int in_size, int out_size, int batch_size, int time_length):
        self.thisptr = new clstm.FwdDeltas(in_size, out_size, batch_size, time_length)

    def __dealloc__(self):
        del self.thisptr

cdef class FwdLayer(object):
    cdef int in_size
    cdef int out_size

    def __cinit__(self, int in_size, int out_size):
        self.in_size = in_size
        self.out_size = out_size

    def get_output_size(self):
        return self.out_size

    def get_input_size(self):
        return self.in_size

    def get_input_buffer_size(self, time_length=1, batch_size=1):
        return self.in_size * time_length * batch_size

    def get_output_buffer_size(self, time_length=1, batch_size=1):
        return self.out_size * time_length * batch_size

    def get_param_size(self, time_length=1, batch_size=1):
        return clstm.FwdWeights(self.in_size, self.out_size).buffer_size()

    def get_internal_state_size(self, time_length=1, batch_size=1):
        return clstm.FwdBuffers(self.in_size, self.out_size, batch_size, time_length).buffer_size()

    def get_internal_error_state_size(self, time_length=1, batch_size=1):
        return clstm.FwdDeltas(self.in_size, self.out_size, batch_size, time_length).buffer_size()

    def create_input_view(self, input_buffer, time_length=1, batch_size=1):
        assert len(input_buffer) == self.get_input_buffer_size(time_length, batch_size)
        return input_buffer.reshape(time_length, batch_size, self.in_size)

    def create_output_view(self, output_buffer, time_length=1, batch_size=1):
        assert len(output_buffer) == self.get_output_buffer_size(time_length, batch_size)
        return output_buffer.reshape(time_length, batch_size, self.out_size)

    def create_param_view(self, BufferView param_buffer, time_length=1, batch_size=1):
        params = FwdParamBuffer(self.in_size, self.out_size)
        #params.thisptr.allocate(param_buffer.flatten2D())
        return params

    def create_internal_view(self, BufferView internal_buffer, time_length=1, batch_size=1):
        internal = FwdInternalBuffer(self.in_size, self.out_size, batch_size, time_length)
        #internal.thisptr.allocate(internal_buffer.flatten2D())
        return internal

    def create_internal_error_view(self, BufferView internal_error_buffer, time_length=1, batch_size=1):
        deltas = FwdParamBuffer(self.in_size, self.out_size, batch_size, time_length)
        #deltas.thisptr.allocate(internal_error_buffer.flatten2D())
        return deltas

    def forward(self, FwdParamBuffer param, FwdInternalBuffer internal, BufferView input, BufferView output):
        clstm.fwd_forward(deref(param.thisptr), deref(internal.thisptr), input.view, output.view)

    def backward(self, FwdParamBuffer param, FwdInternalBuffer internal, FwdErrorBuffer err, BufferView output, BufferView in_deltas, BufferView out_deltas):
        clstm.fwd_backward(deref(param.thisptr), deref(internal.thisptr), deref(err.thisptr), output.view, in_deltas.view, out_deltas.view)

    def gradient(self, FwdParamBuffer param, FwdParamBuffer grad, FwdInternalBuffer internal, FwdErrorBuffer err, BufferView output, BufferView input):
        clstm.fwd_grad(deref(param.thisptr), deref(grad.thisptr), deref(internal.thisptr), deref(err.thisptr), output.view, input.view)

