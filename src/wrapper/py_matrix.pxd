cimport c_matrix
from c_matrix cimport MatrixCPU as CMatrixCPU

cdef class MatrixCPU:
    cdef CMatrixCPU *thisptr      # hold a C++ instance which we're wrapping
