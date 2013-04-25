cimport numpy as np

cimport c_matrix as cm

cdef class Matrix:
    cdef cm.Matrix c_obj
    cdef np.ndarray A
