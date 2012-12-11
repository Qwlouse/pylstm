cimport c_matrix
from c_matrix cimport MatrixCPU as CMatrixCPU
cimport numpy as np

cdef class MatrixCPU:
    cdef CMatrixCPU *thisptr      # hold a C++ instance which we're wrapping
    cdef np.ndarray A
