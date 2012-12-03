#ifndef __MATRIX_H__
#define __MATRIX_H__

enum MatrixState {
	NORMAL,
	TRANSPOSED
};

typedef double d_type;
typedef long int size_type;
typedef d_type* raw_ptr_type;

struct MatrixView;

struct Matrix {
	size_type n_rows;
	size_type n_columns;
	size_type n_slices;
	raw_ptr_type data;

	size_type n_size;

  Matrix(size_type _n_rows, size_type _n_columns, size_type _n_slices);
  virtual ~Matrix(){}
  virtual void allocate(){}
};



struct MatrixView2D {
	MatrixState matrix_state;
	size_type n_rows;
	size_type n_columns;
	raw_ptr_type data;

	size_type stride;
	
	MatrixView2D(MatrixState _matrix_state, size_type _n_rows, size_type _n_columns, raw_ptr_type _data, size_type _stride);


};

struct MatrixView3D {
	MatrixState matrix_state;
	size_type n_rows;
	size_type n_columns;
	size_type n_slices;
	raw_ptr_type data;

	size_type stride;
	
	MatrixView3D(MatrixState _matrix_state, size_type _n_rows, size_type _n_columns, size_type _n_slices, raw_ptr_type _data, size_type _stride);
};

#endif

