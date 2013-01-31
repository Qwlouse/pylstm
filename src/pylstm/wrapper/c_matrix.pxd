from libcpp cimport bool

cdef extern from "matrix_cpu.h":
    cdef cppclass MatrixView2DCPU:
        int n_rows
        int n_columns
        int size

        double& operator[](int)

    cdef cppclass MatrixView3DCPU:
        int n_rows
        int n_columns
        int n_slices
        int size
        double* data

        MatrixView3DCPU()
        MatrixView3DCPU(int, int, int)

        MatrixView2DCPU slice(int)
        MatrixView3DCPU slice(int, int)
        MatrixView2DCPU& flatten()

        void set_data(double* d)
        void print_me()
        double& operator[](int)

    cdef cppclass MatrixCPU:
        int n_rows
        int n_columns
        int n_slices
        int size
        MatrixView2DCPU standard_view_2d
        MatrixView3DCPU standard_view_3d

        MatrixCPU(int, int, int)
        MatrixCPU(double*, int, int, int) #todo use d_type and size_t
        void print_me()
        double& operator[](int)



cdef extern from "matrix_operation_cpu.h":
#    void add(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out)

    void add_into_b(MatrixView2DCPU a, MatrixView2DCPU b)

    void add_scalar(MatrixView2DCPU a, double b)

    void mult(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out)

    void mult_add(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out)

    void dot(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out)

    void dot_add(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out)

    void apply_sigmoid(MatrixView2DCPU a, MatrixView2DCPU out)

    void apply_tanh(MatrixView2DCPU a, MatrixView2DCPU out)

    void apply_tanhx2(MatrixView2DCPU a, MatrixView2DCPU out)

    bool equals(MatrixView2DCPU a, MatrixView2DCPU out)
