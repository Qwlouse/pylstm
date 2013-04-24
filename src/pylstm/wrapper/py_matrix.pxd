cimport numpy as np

cimport c_matrix as cm

cdef class Matrix:
    cdef cm.Matrix view
    cdef np.ndarray A
