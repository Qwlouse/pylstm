cimport numpy as np

cimport c_matrix as cm

cdef class Buffer:
    cdef cm.MatrixCPU *thisptr      # hold a C++ instance which we're wrapping
    cdef np.ndarray A

    cdef cm.MatrixView3DCPU get_standard_view(self)


cdef class BufferView:
    cdef cm.MatrixView3DCPU view
    cdef Buffer B

    cdef cm.MatrixView2DCPU flatten2D(self)
