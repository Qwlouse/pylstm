#include "matrix_operation_cpu.h"
#include <iostream>
#include <algorithm>
#include <math.h>
#include "Core.h"

using namespace std;

///Elementwise add
void add(MatrixView2DCPU &a, MatrixView2DCPU &b, MatrixView2DCPU &out);

///Elementwise multiplication
void dot(MatrixView2DCPU &a, MatrixView2DCPU &b, MatrixView2DCPU &out) {
  ASSERT(a.size == b.size);
  dgbmv(&NO_TRANS, &a.size, &a.size, &diff_zero, &diff_zero, &double_one,
          a.data, &diff_one,
          b.data, &diff_one,
          &double_zero,
          out.data, &diff_one);
}

///Elementwise multiplication and add
void dot_add(MatrixView2DCPU &a, MatrixView2DCPU &b, MatrixView2DCPU &out) {
  ASSERT(a.size == b.size);
  dgbmv(&NO_TRANS, &a.size, &a.size, &diff_zero, &diff_zero, &double_one,
          a.data, &diff_one,
          b.data, &diff_one,
          &double_one,
          out.data, &diff_one);
}

inline double sigmoid(double val) {
  return 1.0 / (1.0 + exp(-val));
}

///Apply sigmoid to all units
void apply_sigmoid(MatrixView2DCPU &a, MatrixView2DCPU &out) {
  transform(a.data, a.data + a.size, out.data, sigmoid);
}

///Apply tanh to all units
void apply_tanh(MatrixView2DCPU &a, MatrixView2DCPU &out) {
  transform(a.data, a.data + a.size, out.data, tanh);
}


void mult(MatrixView2DCPU &a, MatrixView2DCPU &b, MatrixView2DCPU &out) {
	char a_state = (a.state == NORMAL) ? 'N' : 'T';
	char b_state = (b.state == NORMAL) ? 'N' : 'T';
    
    cout << a_state << " " << b_state << " " << a.n_rows << " " << b.n_rows << " " << a.n_columns << " " << b.n_columns << " " << a.data << " " << b.data << endl;

	dgemm(&a_state, &b_state, &a.n_rows, &b.n_columns, &a.n_columns,
		&double_one,
		a.data,
		&a.n_rows, b.data, &b.n_rows, &double_zero, out.data, &out.n_rows);
}


void mult_add(MatrixView2DCPU &a, MatrixView2DCPU &b, MatrixView2DCPU &out) {
	char a_state = (a.state == NORMAL) ? 'N' : 'T';
	char b_state = (b.state == NORMAL) ? 'N' : 'T';
    
    cout << a_state << " " << b_state << " " << a.n_rows << " " << b.n_rows << " " << a.n_columns << " " << b.n_columns << " " << a.data << " " << b.data << endl;

	dgemm(&a_state, &b_state, &a.n_rows, &b.n_columns, &a.n_columns,
		&double_one,
		a.data,
		&a.n_rows, b.data, &b.n_rows, &double_one, out.data, &out.n_rows);
}
