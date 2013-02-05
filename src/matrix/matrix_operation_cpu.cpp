#include "matrix_operation_cpu.h"
#include <iostream>
#include <algorithm>
#include <math.h>

#include "Core.h"

using namespace std;

static d_type double_one = 1.0;
static d_type double_zero = 0.0;
//static d_type double_min_one = -1.0;
static char NO_TRANS = 'N';
//static char TRANS = 'T';
static ptrdiff_t diff_one = 1;
static ptrdiff_t diff_zero = 0;

void add_into_b(MatrixView2DCPU a, MatrixView2DCPU b) {
  //size_type len(a.size);
  daxpy(&a.size, &double_one, a.data, &diff_one, b.data, &diff_one);
}

/*
///Elementwise add
void add(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out) {
  if (a.data == out.data) {
  } else if() {
    
  }
}
*/

void add_scalar(MatrixView2DCPU a, d_type b) {
  //MatrixView2DCPU::iterator it(arg1.begin());
  //	MatrixView2DCPU::iterator end(arg1.end());
  //for (;it != end; ++it) *it += arg2;
  daxpy(&a.size, &double_one, &b, &diff_zero, a.data, &diff_one);
}


///Copy stuff
void copy(MatrixView2DCPU a, MatrixView2DCPU b) {
  ASSERT(a.size == b.size);
  ptrdiff_t ridiculous(a.size);
  dcopy(&ridiculous, a.data, &diff_one, b.data, &diff_one);
}

void copy(MatrixView3DCPU a, MatrixView3DCPU b) {
  ASSERT(a.size == b.size);
  ptrdiff_t ridiculous(a.size);
  dcopy(&ridiculous, a.data, &diff_one, b.data, &diff_one);
}

///Elementwise multiplication
void dot(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out) {
  ASSERT(a.size == b.size);
  ASSERT(a.state == NORMAL && b.state == NORMAL && out.state == NORMAL);

  dgbmv(&NO_TRANS, &a.size, &a.size, &diff_zero, &diff_zero, &double_one,
          a.data, &diff_one,
          b.data, &diff_one,
          &double_zero,
          out.data, &diff_one);
}

///Elementwise multiplication and add
void dot_add(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out) {
  ASSERT(a.size == b.size);
  ASSERT(a.state == NORMAL && b.state == NORMAL && out.state == NORMAL);

  dgbmv(&NO_TRANS, &a.size, &a.size, &diff_zero, &diff_zero, &double_one,
          a.data, &diff_one,
          b.data, &diff_one,
          &double_one,
          out.data, &diff_one);
}

///Elementwise multiplication and add, with squash to size of out (out is smaller than a and b)
void dot_add_squash(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out, d_type const scale) {
  ASSERT(a.size == b.size);
  ASSERT(a.size % out.size == 0);
  ASSERT(a.state == NORMAL && b.state == NORMAL && out.state == NORMAL);
  
  raw_ptr_type a_i(a.data), b_i(b.data), a_end(a.data + a.size);
  raw_ptr_type out_start(out.data), out_i(out.data), out_end(out.data + out.size);

  for (; a_i != a_end; ++a_i, ++b_i, ++out_i) {
    if (out_i == out_end)
      out_i = out_start;
    *out_i += *a_i * *b_i * scale;
  }
}


///Elementwise multiplication, with squash to size of out (out is smaller than a and b)
void dot_squash(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out, d_type const scale) {
  ASSERT(a.size == b.size);
  ASSERT(a.size % out.size == 0);
  ASSERT(a.state == NORMAL && b.state == NORMAL && out.state == NORMAL);
  
  raw_ptr_type a_i(a.data), b_i(b.data), a_end(a.data + a.size);
  raw_ptr_type out_start(out.data), out_i(out.data), out_end(out.data + out.size);

  fill(out_start, out_end, 0.0);

  for (; a_i != a_end; ++a_i, ++b_i, ++out_i) {
    if (out_i == out_end)
      out_i = out_start;
    *out_i += *a_i * *b_i * scale;
  }
  if (scale != 1.0) {
    out_i = out_start;
    for (; out_i != out_start; ++out_i)
      *out_i *= scale;
  }
}

void squash(MatrixView2DCPU a, MatrixView2DCPU out, d_type const scale) {
  raw_ptr_type a_i(a.data), a_end(a.data + a.size);
  raw_ptr_type out_start(out.data), out_i(out.data), out_end(out.data + out.size);

  fill(out_start, out_end, 0.0);

  for (; a_i != a_end; ++a_i, ++out_i) {
    if (out_i == out_end)
      out_i = out_start;
    *out_i += *a_i;
  }

  if (scale != 1.0) {
    out_i = out_start;
    for (; out_i != out_start; ++out_i)
      *out_i *= scale;
  }
}

inline double sigmoid(double val) {
  return 1.0 / (1.0 + exp(-val));
}

inline double tanhx2(double val) {
  return 2.0 * tanh(val);
}

inline double tanh_(double val) {
  return tanh(val);
}

///Apply sigmoid to all units
void apply_sigmoid(MatrixView2DCPU a, MatrixView2DCPU out) {
  transform(a.data, a.data + a.size, out.data, sigmoid);
}

///Apply tanh to all units
void apply_tanh(MatrixView2DCPU a, MatrixView2DCPU out) {
  transform(a.data, a.data + a.size, out.data, tanh_);
}

///Apply tanh * 2to all units
void apply_tanhx2(MatrixView2DCPU a, MatrixView2DCPU out) {
  transform(a.data, a.data + a.size, out.data, tanhx2);
}


void mult(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out, d_type scale) {
	char a_state = (a.state == NORMAL) ? 'N' : 'T';
	char b_state = (b.state == NORMAL) ? 'N' : 'T';
    
    cout << a_state << " " << b_state << " " << a.n_rows << " " << b.n_rows << " " << a.n_columns << " " << b.n_columns << " " << a.data << " " << b.data << endl;

	dgemm(&a_state, &b_state, &a.n_rows, &b.n_columns, &a.n_columns,
		&scale,
		a.data,
		&a.n_rows, b.data, &b.n_rows, &double_zero, out.data, &out.n_rows);
}


void mult_add(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out, d_type scale) {
	char a_state = (a.state == NORMAL) ? 'N' : 'T';
	char b_state = (b.state == NORMAL) ? 'N' : 'T';
    
    cout << a_state << " " << b_state << " " << a.n_rows << " " << b.n_rows << " " << a.n_columns << " " << b.n_columns << " " << a.data << " " << b.data << endl;

	dgemm(&a_state, &b_state, &a.n_rows, &b.n_columns, &a.n_columns,
		&scale,
		a.data,
		&a.n_rows, b.data, &b.n_rows, &double_one, out.data, &out.n_rows);
}

void mult_vector(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out) {
	char a_state = (a.state == NORMAL) ? 'N' : 'T';
	char b_state = (b.state == NORMAL) ? 'N' : 'T';
    
    cout << a_state << " " << b_state << " " << a.n_rows << " " << b.n_rows << " " << a.n_columns << " " << b.n_columns << " " << a.data << " " << b.data << endl;

	dgemm(&a_state, &b_state, &a.n_rows, &b.n_columns, &a.n_columns,
		&double_one,
		a.data,
		&a.n_rows, b.data, &b.n_rows, &double_zero, out.data, &out.n_rows);
}

bool equals(MatrixView2DCPU a, MatrixView2DCPU b) {
  if (a.n_rows != b.n_rows || a.n_columns != b.n_columns)
    return false;
  for (size_t i(0); i < a.size; ++i)
    if (a.data[i] != b.data[i])
      return false;
  return true;
}

