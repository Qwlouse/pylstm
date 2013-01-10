#cimport c_matrix as cm
cimport c_lstm_layer as clstm
cimport numpy as np

cdef class LstmLayer:
    cdef int in_size
    cdef int out_size


