cimport c_matrix as cm
cimport numpy as np

cdef class MatrixCPU:
    cdef cm.MatrixCPU *thisptr      # hold a C++ instance which we're wrapping
    cdef np.ndarray A
    cdef inline cm.MatrixView2DCPU get_2d_view(self):
        return self.thisptr.standard_view_2d

    cdef inline cm.MatrixView3DCPU get_3d_view(self):
        return self.thisptr.standard_view_3d
