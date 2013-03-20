#include "matrix_operation.h"

#include <algorithm>
#include <iostream>
#include <cmath>
#include <string>

#include "Core.h"

using namespace std;

static d_type double_one = 1.0;
static d_type double_zero = 0.0;
//static d_type double_min_one = -1.0;
static char NO_TRANS = 'N';
//static char TRANS = 'T';
static ptrdiff_t diff_one = 1;
static ptrdiff_t diff_zero = 0;

bool equals(Matrix a, Matrix b) {
	if (a.n_rows != b.n_rows || a.n_columns != b.n_columns || a.n_slices != b.n_slices) {
		return false;
	}
        
	for (size_t slice = 0; slice < a.n_slices; ++slice ) {
		for (size_t column = 0; column < a.n_columns; ++column ) {
			for (size_t row = 0; row < a.n_rows; ++row ) {
				if (a.get(row, column, slice) != b.get(row, column, slice)) {
					return false;
				}
			}
		}
	}
    return true;
}

void add_into_b(Matrix a, Matrix b) {
    //size_type len(a.size);
    // daxpy(n, a, x, incx, y, incy)
    // for i = 0 to n : y[i*incy] += a * x[i*incx]
	//
	ASSERT(a.state == b.state);
	long int n = a.size;
    daxpy(&n, &double_one, a.get_data(), &diff_one, b.get_data(), &diff_one);
}

void add_vector_into(Matrix vec, Matrix mat) {
  size_t j = 0;
  for (int i = 0; i < mat.size; ++i, ++j) {
	  if (j == vec.size) {
		  j = 0;
	  }
	  mat[i] += vec[j];
  }
}


void add_scalar(Matrix a, d_type b) {
	//MatrixView2DCPU::iterator it(arg1.begin());
	//	MatrixView2DCPU::iterator end(arg1.end());
	//for (;it != end; ++it) *it += arg2;
	long int n = a.size;
	daxpy(&n, &double_one, &b, &diff_zero, a.get_data(), &diff_one);
}

///Elementwise multiplication
void dot(Matrix a, Matrix b, Matrix out) {
	ASSERT(a.size == out.size);
	ASSERT(a.size >= b.size);
	ASSERT(a.size % b.size == 0);

	ASSERT(a.state == NORMAL && b.state == NORMAL && out.state == NORMAL);

	fill(out.get_data(), out.get_data() + out.size, 0.0);

	if (a.size == b.size) {
		long int n = a.size;
		dgbmv(&NO_TRANS, &n, &n, &diff_zero, &diff_zero, &double_one,
				a.get_data(), &diff_one,
				b.get_data(), &diff_one,
				&double_one,
				out.get_data(), &diff_one);
	} else {
		d_type* a_i(a.get_data());
		d_type* a_end(a.get_data() + a.size);
		d_type* out_i(out.get_data());
		d_type* b_start(b.get_data());
		d_type* b_i(b.get_data());
		d_type* b_end(b.get_data() + b.size);

		for (; a_i != a_end; ++a_i, ++b_i, ++out_i) {
			if (b_i == b_end)
				b_i = b_start;
			*out_i += *a_i * *b_i;
		}

	}
}

void dot_into_b(Matrix a, Matrix b) {
    ASSERT(a.size == b.size);
    for (size_t i = 0; i < a.size; ++i) {
        b[i] *= a[i];
    }
}


///Elementwise multiplication and add
void dot_add(Matrix a, Matrix b, Matrix out) {
	ASSERT(a.size == out.size);
	ASSERT(a.size >= b.size);
	ASSERT(a.size % b.size == 0);

	ASSERT(a.state == NORMAL && b.state == NORMAL && out.state == NORMAL);

	if (a.size == b.size) {
		long int n = a.size;
		dgbmv(&NO_TRANS, &n, &n, &diff_zero, &diff_zero, &double_one,
				a.get_data(), &diff_one,
				b.get_data(), &diff_one,
				&double_one,
				out.get_data(), &diff_one);
	} else {
		d_type* a_i(a.get_data());
		d_type* a_end(a.get_data() + a.size);
		d_type* out_i(out.get_data());
		d_type* b_start(b.get_data());
		d_type* b_i(b.get_data());
		d_type* b_end(b.get_data() + b.size);

		for (; a_i != a_end; ++a_i, ++b_i, ++out_i) {
			if (b_i == b_end)
				b_i = b_start;
			*out_i += *a_i * *b_i;
		}

	}
}

void mult(Matrix a, Matrix b, Matrix out, d_type scale) {
  char a_state = (a.state == NORMAL) ? 'N' : 'T';
  char b_state = (b.state == NORMAL) ? 'N' : 'T';
  ASSERT(out.state == NORMAL);
  ASSERT(a.n_slices == 1);
  ASSERT(b.n_slices == 1);
  ASSERT(out.n_slices == 1);
  ASSERT(a.n_columns == b.n_rows);
  ASSERT(out.n_rows == a.n_rows);
  ASSERT(out.n_columns == b.n_columns);

  ptrdiff_t M = a.n_rows;
  ptrdiff_t N = b.n_columns;
  ptrdiff_t K = a.n_columns;

  // the size of the first dimension of the matrices, as laid out in memory;
  // meaning the memory distance between the start of each row/column,
  // depending on the memory structure
  ptrdiff_t a_stride = a.state == NORMAL ? a.n_rows : a.n_columns;
  ptrdiff_t b_stride = b.state == NORMAL ? b.n_rows : b.n_columns;
  ptrdiff_t out_stride = out.n_rows;

  dgemm(&a_state, &b_state, &M, &N, &K, &scale, a.get_data(),
	&a_stride, b.get_data(), &b_stride, &double_zero, out.get_data(), &out_stride);
}

void mult_add(Matrix a, Matrix b, Matrix out, d_type scale) {
	char a_state = (a.state == NORMAL) ? 'N' : 'T';
	char b_state = (b.state == NORMAL) ? 'N' : 'T';
	ASSERT(out.state == NORMAL);
    ASSERT(a.n_slices == 1);
    ASSERT(b.n_slices == 1);
    ASSERT(out.n_slices == 1);
    ASSERT(a.n_columns == b.n_rows);
    ASSERT(out.n_rows == a.n_rows);
    ASSERT(out.n_columns == b.n_columns);

    ptrdiff_t M = a.n_rows;
    ptrdiff_t N = b.n_columns;
    ptrdiff_t K = a.n_columns;

    // the size of the first dimension of the matrices, as laid out in memory;
    // meaning the memory distance between the start of each row/column,
    // depending on the memory structure
    ptrdiff_t a_stride = a.state == NORMAL ? a.n_rows : a.n_columns;
    ptrdiff_t b_stride = b.state == NORMAL ? b.n_rows : b.n_columns;
    ptrdiff_t out_stride = out.n_rows;

	//cout << a_state << " " << b_state << " " << a.n_rows << " " << b.n_rows << " " << a.n_columns << " " << b.n_columns << " " << a.data << " " << b.data << endl;
	ptrdiff_t a_rows = a.n_rows;
	ptrdiff_t a_columns = a.n_columns;
	ptrdiff_t b_rows = b.n_rows;
	ptrdiff_t b_columns = b.n_columns;
	ptrdiff_t out_rows = out.n_rows;
	dgemm(&a_state, &b_state, &M, &N, &K, &scale, a.get_data(),
	      &a_stride, b.get_data(), &b_stride, &double_one, out.get_data(), &out_stride);
}


void ActivationFunction::apply(Matrix in, Matrix out) const {
    transform(in.get_data(), in.get_data() + in.size, out.get_data(), *f);
}

void ActivationFunction::apply_deriv(Matrix in, Matrix d, Matrix out) const {
    transform(in.get_data(), in.get_data() + in.size, out.get_data(), *deriv);
    dot_into_b(d, out);
}

// Softmax Layer works slightly differently
void SoftmaxLayerActivation::apply(Matrix in, Matrix out) const {
  for (size_t slice = 0; slice < in.n_slices; ++slice ) {
    for (size_t column = 0; column < in.n_columns; ++column ) {
        // determine the max
        d_type col_max = 0;
        for (size_t row = 0; row < in.n_rows; ++row ) {
            col_max = std::max(col_max, in.get(row, column, slice));
        }

        d_type col_sum = 0.0;
        for (size_t row = 0; row < in.n_rows; ++row ) {
            // trick to avoid numerical instability
            out.get(row, column, slice) = exp(in.get(row, column, slice) - col_max);
            col_sum += out.get(row, column, slice);
        }
        for (size_t row = 0; row < in.n_rows; ++row ) {
            out.get(row, column, slice) = out.get(row, column, slice)/col_sum;
        }
    }
  }
}

/*
 The derivative is more involved for the general case
 For i'th unit with input x_i, output y_i, and Error E from next layer
 dE/dx_i = y_i * (dE/dy_i - sum(dE/dy_j * y_j))
*/
void SoftmaxLayerActivation::apply_deriv(Matrix in, Matrix d, Matrix out) const {
  for (size_t slice = 0; slice < in.n_slices; ++slice ) {
    for (size_t column = 0; column < in.n_columns; ++column ) {
      // Calculate the attenuation term
      d_type delta_attenuation = 0.0;
      for (size_t row = 0; row < in.n_rows; ++row) {
        delta_attenuation += d.get(row, column, slice) * in.get(row, column, slice);
      }
      for (size_t row = 0; row < in.n_rows; ++row) {
        out.get(row, column, slice) = in.get(row, column, slice) * (d.get(row, column, slice) - delta_attenuation);
      }
    }
  }
}


void apply(Matrix in, Matrix out, unary_double_func f) {
	transform(in.get_data(), in.get_data() + in.size, out.get_data(), *f);
}

///Apply sigmoid to all units
void apply_sigmoid(Matrix a, Matrix out) {
  transform(a.get_data(), a.get_data() + a.size, out.get_data(), sigmoid);
}

void apply_sigmoid_deriv(Matrix a, Matrix out) {
  transform(a.get_data(), a.get_data() + a.size, out.get_data(), sigmoid_deriv);
}

///Apply tanh to all units
void apply_tanh(Matrix a, Matrix out) {
  transform(a.get_data(), a.get_data() + a.size, out.get_data(), tanh_);
}

void apply_tanh_deriv(Matrix a, Matrix out) {
  transform(a.get_data(), a.get_data() + a.size, out.get_data(), tanh_deriv);
}

///Apply tanh * 2to all units
void apply_tanhx2(Matrix a, Matrix out) {
  transform(a.get_data(), a.get_data() + a.size, out.get_data(), tanhx2);
}

void apply_tanhx2_deriv(Matrix a, Matrix out) {
  transform(a.get_data(), a.get_data() + a.size, out.get_data(), tanhx2_deriv);
}

///Copy the data of one matrix into another
void copy(Matrix a, Matrix b) {
  ASSERT(a.size == b.size);
  ptrdiff_t ridiculous(a.size);
  dcopy(&ridiculous, a.get_data(), &diff_one, b.get_data(), &diff_one);
}

void squash(Matrix a, Matrix out) {
  out.set_all_elements_to(0.0);
  int out_index = 0;
  for (int i=0; i < a.size; ++i, ++out_index) {
    if (out_index == out.size)
      out_index = 0;
    out[out_index] += a[i];
  }
}


///Elementwise multiplication, with squash to size of out (out is smaller than a and b)
void dot_squash(Matrix a, Matrix b, Matrix out) {
  ASSERT(a.size == b.size);
  ASSERT(a.size % out.size == 0);
  ASSERT(a.state == b.state && b.state == out.state);

  out.set_all_elements_to(0.0);
  int out_index = 0;
    for (int i=0; i < a.size; ++i, ++out_index) {
      if (out_index == out.size)
    	  out_index = 0;
      out[out_index] += a[i] * b[i];
    }
}

///scale matrix by a scalar
void scale_into(Matrix a, d_type alpha) {
  long int len(a.size);
  dscal(&len, &alpha, a.get_data(), &diff_one);
}


/*
///Elementwise add
void add(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out) {
  if (a.data == out.data) {
  } else if() {

  }
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



void mult_vector(MatrixView2DCPU a, MatrixView2DCPU b, MatrixView2DCPU out) {
  char a_state = (a.state == NORMAL) ? 'N' : 'T';
  char b_state = (b.state == NORMAL) ? 'N' : 'T';
  
  //cout << a_state << " " << b_state << " " << a.n_rows << " " << b.n_rows << " " << a.n_columns << " " << b.n_columns << " " << a.data << " " << b.data << endl;
  
  dgemm(&a_state, &b_state, &a.n_rows, &b.n_columns, &a.n_columns,
	&double_one,
	a.data,
	&a.n_rows, b.data, &b.n_rows, &double_zero, out.data, &out.n_rows);
}


*/
