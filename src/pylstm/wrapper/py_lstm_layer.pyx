#!/usr/bin/python
# coding=utf-8
cimport c_lstm_layer as clstm
cimport c_matrix as cm
from cython.operator cimport dereference as deref
from py_matrix cimport BufferView

cdef class LstmParamBuffer(object):
    cdef clstm.LstmWeights* thisptr
    def __cinit__(self, int in_size, int out_size):
        self.thisptr = new clstm.LstmWeights(in_size, out_size)

    cdef BufferView create_BufferView_from_2dview(self, cm.MatrixView2DCPU view):
        buffer_view = BufferView()
        buffer_view.view = cm.MatrixView3DCPU(view.n_rows, view.n_columns, 1)
        buffer_view.view.set_data(&view[0])
        return buffer_view

    def get_IX(self):
        return self.create_BufferView_from_2dview(self.thisptr.IX)

    def get_IH(self):
        return self.create_BufferView_from_2dview(self.thisptr.IH)

    def get_IS(self):
        return self.create_BufferView_from_2dview(self.thisptr.IS)

    def get_FX(self):
        return self.create_BufferView_from_2dview(self.thisptr.FX)

    def get_FH(self):
        return self.create_BufferView_from_2dview(self.thisptr.FH)

    def get_FS(self):
        return self.create_BufferView_from_2dview(self.thisptr.FS)

    def get_ZH(self):
        return self.create_BufferView_from_2dview(self.thisptr.ZH)

    def get_ZX(self):
        return self.create_BufferView_from_2dview(self.thisptr.ZX)

    def get_OX(self):
        return self.create_BufferView_from_2dview(self.thisptr.OX)

    def get_OH(self):
        return self.create_BufferView_from_2dview(self.thisptr.OH)

    def get_OS(self):
        return self.create_BufferView_from_2dview(self.thisptr.OS)

    def get_I_bias(self):
        return self.create_BufferView_from_2dview(self.thisptr.I_bias)

    def get_F_bias(self):
        return self.create_BufferView_from_2dview(self.thisptr.F_bias)

    def get_Z_bias(self):
        return self.create_BufferView_from_2dview(self.thisptr.Z_bias)

    def get_O_bias(self):
        return self.create_BufferView_from_2dview(self.thisptr.O_bias)




    def __dealloc__(self):
        del self.thisptr

cdef class LstmInternalBuffer(object):
    cdef clstm.LstmBuffers* thisptr
    def __cinit__(self, int in_size, int out_size, int batch_size, int time_length):
        self.thisptr = new clstm.LstmBuffers(in_size, out_size, batch_size, time_length)

    def __dealloc__(self):
        del self.thisptr

cdef class LstmErrorBuffer(object):
    cdef clstm.LstmDeltas* thisptr
    def __cinit__(self, int in_size, int out_size, int batch_size, int time_length):
        self.thisptr = new clstm.LstmDeltas(in_size, out_size, batch_size, time_length)

    def __dealloc__(self):
        del self.thisptr

cdef class LstmLayer(object):
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
        return clstm.LstmWeights(self.in_size, self.out_size).buffer_size()

    def get_internal_state_size(self, time_length=1, batch_size=1):
        return clstm.LstmBuffers(self.in_size, self.out_size, batch_size, time_length).buffer_size()

    def get_internal_error_state_size(self, time_length=1, batch_size=1):
        return clstm.LstmDeltas(self.in_size, self.out_size, batch_size, time_length).buffer_size()

    def create_input_view(self, input_buffer, time_length=1, batch_size=1):
        assert len(input_buffer) == self.get_input_buffer_size(time_length, batch_size)
        return input_buffer.reshape(time_length, batch_size, self.in_size)

    def create_output_view(self, output_buffer, time_length=1, batch_size=1):
        assert len(output_buffer) == self.get_output_buffer_size(time_length, batch_size)
        return output_buffer.reshape(time_length, batch_size, self.out_size)

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

    def gradient(self, LstmParamBuffer param, LstmParamBuffer grad, LstmInternalBuffer internal, LstmErrorBuffer err, BufferView output, BufferView input):
        clstm.lstm_grad(deref(param.thisptr), deref(grad.thisptr), deref(internal.thisptr), deref(err.thisptr), output.view, input.view)

    def r_pass(self, LstmParamBuffer param, LstmParamBuffer v, LstmInternalBuffer internal, LstmInternalBuffer r_internal, BufferView input_view, BufferView out, BufferView r_out):
        clstm.lstm_Rpass(deref(param.thisptr), deref(v.thisptr),  deref(internal.thisptr), deref(r_internal.thisptr), input_view.view, out.view, r_out.view)

    #def r_backward(self, LstmParamBuffer param, LstmInternalBuffer internal, LstmErrorBuffer err, BufferView out, BufferView in_deltas, BufferView out_deltas, LstmInternalBuffer r_internal, float _lambda, float mu):
        #clstm.lstm_Rbackward(deref(param.thisptr), deref(internal.thisptr), deref(err.thisptr), out.view, in_deltas.view, out_deltas.view, deref(r_internal.thisptr), _lambda, mu)
    def r_backward(self, LstmParamBuffer param, LstmInternalBuffer internal, LstmErrorBuffer err, BufferView in_deltas, BufferView out_deltas, LstmInternalBuffer r_internal, float _lambda, float mu):
        clstm.lstm_Rbackward(deref(param.thisptr), deref(internal.thisptr), deref(err.thisptr), in_deltas.view, out_deltas.view, deref(r_internal.thisptr), _lambda, mu)


