#cimport c_matrix as cm
cimport c_lstm_layer as clstm
cimport numpy as np
cimport py_matrix as pm
from cython.operator cimport dereference as deref

cdef class LstmParamBuffer:
    cdef clstm.LstmWeights* thisptr

cdef class LstmInternalBuffer:
    cdef clstm.LstmBuffers* thisptr

cdef class LstmErrorBuffer:
    cdef clstm.LstmDeltas* thisptr


cdef class LstmLayer:
    cdef int in_size
    cdef int out_size

    cdef inline int get_weight_size(self):
        return clstm.LstmWeights(self.in_size, self.out_size).buffer_size()

cdef inline void lstm_forward(pm.MatrixCPU input, LstmParamBuffer param, LstmInternalBuffer internal, pm.MatrixCPU output):
    clstm.lstm_forward(deref(param.thisptr), deref(internal.thisptr), input.get_3d_view(), output.get_3d_view())
