#ifndef __MATRIX_CPU_H__
#define __MATRIX_CPU_H__

#include "matrix.h"
#include <iostream>

struct MatrixView2DCPU : public MatrixView2D {
  MatrixView2DCPU(MatrixState _matrix_state, size_type _n_rows, size_type _n_columns, raw_ptr_type _data, size_type _stride);

  MatrixView2DCPU();
};


struct MatrixView3DCPU : public MatrixView3D {
  MatrixView3DCPU();
  MatrixView3DCPU(MatrixState _matrix_state, size_type _n_rows, size_type _n_columns, size_type _n_slices, raw_ptr_type _data, size_type _stride);

  MatrixView2DCPU &flatten();
  MatrixView2DCPU slice(size_type t);

  MatrixView2DCPU matrix_view_2d;
};

struct MatrixCPU : public Matrix {
    bool owns_data;
	MatrixView2DCPU standard_view_2d;
	MatrixView3DCPU standard_view_3d;

	MatrixCPU(size_type _n_rows, size_type _n_columns, size_type _n_slices);
    MatrixCPU(std::initializer_list<std::initializer_list<double>> values);
    MatrixCPU(d_type* _data, size_type _n_rows, size_type _n_columns, size_type _n_slices);
    virtual ~MatrixCPU();

	operator MatrixView2DCPU&() {return standard_view_2d;}
	operator MatrixView3DCPU&() {return standard_view_3d;}

	virtual void allocate();
	void print_me();
};



std::ostream &operator<<(std::ostream &out, MatrixCPU &in);


#endif
