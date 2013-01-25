#ifndef __MATRIX_CPU_H__
#define __MATRIX_CPU_H__

#include "matrix.h"
#include <iostream>
#include <vector>
#include <initializer_list>


struct MatrixView2DCPU : public MatrixView2D {
  MatrixView2DCPU();
  MatrixView2DCPU(MatrixState _matrix_state, size_type _n_rows, size_type _n_columns, raw_ptr_type _data, size_type _stride = 0);
  MatrixView2DCPU(size_type _n_rows, size_type _n_columns);
  void print_me();
  MatrixView2DCPU T();

  void set_data(raw_ptr_type d) {data = d;}
};


struct MatrixView3DCPU : public MatrixView3D {
  MatrixView3DCPU();
  MatrixView3DCPU(MatrixState _matrix_state, size_type _n_rows, size_type _n_columns, size_type _n_slices, raw_ptr_type _data, size_type _stride);
  MatrixView3DCPU(size_type _n_rows, size_type _n_columns, size_type _n_slices);

  MatrixView2DCPU &flatten();
  MatrixView2DCPU slice(size_type t);
  MatrixView3DCPU slice(size_type start, size_type stop);

  void set_data(raw_ptr_type d) {data = d; matrix_view_2d.set_data(d);}

  MatrixView2DCPU matrix_view_2d;

  MatrixView3DCPU subslice(size_type start, size_type end);

  operator MatrixView2DCPU() { return flatten(); } 

  void print_me();

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


void lay_out(MatrixView2DCPU &buffer_view, std::vector<MatrixView2DCPU*> &buffers);
void lay_out(MatrixView2DCPU &buffer_view, std::vector<MatrixView3DCPU*> &buffers);

std::ostream &operator<<(std::ostream &out, MatrixCPU &in);



#endif

