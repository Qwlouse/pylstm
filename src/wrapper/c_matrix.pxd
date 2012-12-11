cdef extern from "matrix_cpu.h":
    cdef cppclass MatrixCPU:
        MatrixCPU(int, int, int)
        MatrixCPU(double*, int, int, int) #todo use d_type and size_t


