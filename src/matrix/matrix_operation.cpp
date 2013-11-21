#include "matrix_operation.h"

#include <algorithm>
#include <iostream>

#include "Core.h"
#include "cblas_wrapper.h"


using namespace std;

bool equals(Matrix a, Matrix b) {
	if (a.n_rows != b.n_rows || a.n_columns != b.n_columns || a.n_slices != b.n_slices) {
		return false;
	}

    for (auto ita = a.begin(), itb = b.begin(); ita != a.end(); ++ita, ++itb) {
        if (*ita != *itb) {
            return false;
        }
    }
    return true;
}

void add_into_b(Matrix a, Matrix b) {
	if (a.state == b.state && a.stride == 0 && b.stride == 0) {
        cblas_daxpy(static_cast<int>(a.size), 1.0, a.get_data(), 1, b.get_data(), 1);
    } else {
        Matrix::iterator ita = a.begin();
        Matrix::iterator ita_end = a.end();
        Matrix::iterator itb = b.begin();
        for (; ita != ita_end; ++ita, ++itb) {
            *itb += *ita;
        }
    }
}

void add_vector_into(Matrix vec, Matrix mat) {
  auto it_v_end = vec.end();
  for (auto it = mat.begin(), it_v = vec.begin(); it != mat.end(); ++it, ++it_v) {
	  if (it_v == it_v_end) {
		  it_v = vec.begin();
	  }
	  *it += *it_v;
  }
}


void add_scalar(Matrix a, d_type b) {
	if (a.stride == 0) {
	    cblas_daxpy(static_cast<int>(a.size), 1.0, &b, 0, a.get_data(), 1);
	} else {
	    for (auto it = a.begin(); it != a.end(); ++it) {
	        *it += b;
	    }
	}
}

///Elementwise multiplication
void dot(Matrix a, Matrix b, Matrix out) {
	ASSERT(a.size == out.size);
	ASSERT(a.size >= b.size);
	ASSERT(a.size % b.size == 0);

	ASSERT(a.state == NORMAL && b.state == NORMAL && out.state == NORMAL);

	if (a.size == b.size && a.stride == 0 && b.stride == 0 && out.stride == 0) {
		int n = static_cast<int>(a.size);
		cblas_dgbmv(CblasColMajor, CblasNoTrans, n, n, 0, 0, 1.0,
				a.get_data(), 1,
				b.get_data(), 1,
				0.0,
				out.get_data(), 1);
	} else {

	    fill(out.begin(), out.end(), 0.0);
	    auto a_end = a.end();
	    auto b_end = b.end();

		for (auto ita=a.begin(), itb=b.begin(), ito=out.begin(); ita != a_end; ++ita, ++itb, ++ito) {
			if (itb == b_end)
				itb = b.begin();
			*ito += *ita * *itb;
		}

	}
}

void dot_into_b(Matrix a, Matrix b) {
    ASSERT(a.size == b.size);
    auto enda = a.end();
    for (auto ita = a.begin(), itb = b.begin(); ita != enda; ++ita, ++itb) {
        *itb *= *ita;
    }
}


///Elementwise multiplication and add
void dot_add(Matrix a, Matrix b, Matrix out) {
	ASSERT(a.size == out.size);
	ASSERT(a.size >= b.size);
	ASSERT(a.size % b.size == 0);

	if (a.state == NORMAL && b.state == NORMAL && out.state == NORMAL && a.size == b.size && a.stride == 0 && b.stride == 0 && out.stride == 0) {
		int n = static_cast<int>(a.size);
		cblas_dgbmv(CblasColMajor, CblasNoTrans, n, n, 0, 0, 1.0,
				a.get_data(), 1,
				b.get_data(), 1,
				1.0,
				out.get_data(), 1);
	} else {
		auto a_end = a.end();
        auto b_end = b.end();

        for (auto ita=a.begin(), itb=b.begin(), ito=out.begin(); ita != a_end; ++ita, ++itb, ++ito) {
            if (itb == b_end)
                itb = b.begin();
            *ito += *ita * *itb;
        }
	}
}

void mult(Matrix a, Matrix b, Matrix out, d_type scale) {
    enum CBLAS_TRANSPOSE a_state = (a.state == NORMAL) ? CblasNoTrans : CblasTrans;
    enum CBLAS_TRANSPOSE b_state = (b.state == NORMAL) ? CblasNoTrans : CblasTrans;
    ASSERT(out.state == NORMAL);
    ASSERT(a.n_slices == 1);
    ASSERT(b.n_slices == 1);
    ASSERT(out.n_slices == 1);
    ASSERT(a.n_columns == b.n_rows);
    ASSERT(out.n_rows == a.n_rows);
    ASSERT(out.n_columns == b.n_columns);

    int M = static_cast<int>(a.n_rows);
    int N = static_cast<int>(b.n_columns);
    int K = static_cast<int>(a.n_columns);

    // the size of the first dimension of the matrices, as laid out in memory;
    // meaning the memory distance between the start of each row/column,
    // depending on the memory structure
    int a_stride = static_cast<int>(a.state == NORMAL ? a.n_rows+a.stride : a.n_columns+a.stride);
    int b_stride = static_cast<int>(b.state == NORMAL ? b.n_rows+b.stride : b.n_columns+b.stride);
    int out_stride = static_cast<int>(out.n_rows+out.stride);

    cblas_dgemm(CblasColMajor, a_state, b_state, M, N, K, scale, a.get_data(),
	    a_stride, b.get_data(), b_stride, 0.0, out.get_data(), out_stride);
}

void mult_add(Matrix a, Matrix b, Matrix out, d_type scale) {
	enum CBLAS_TRANSPOSE a_state = (a.state == NORMAL) ? CblasNoTrans : CblasTrans;
    enum CBLAS_TRANSPOSE b_state = (b.state == NORMAL) ? CblasNoTrans : CblasTrans;
    ASSERT(out.state == NORMAL);
    ASSERT(a.n_slices == 1);
    ASSERT(b.n_slices == 1);
    ASSERT(out.n_slices == 1);
    ASSERT(a.n_columns == b.n_rows);
    ASSERT(out.n_rows == a.n_rows);
    ASSERT(out.n_columns == b.n_columns);

    int M = static_cast<int>(a.n_rows);
    int N = static_cast<int>(b.n_columns);
    int K = static_cast<int>(a.n_columns);

    // the size of the first dimension of the matrices, as laid out in memory;
    // meaning the memory distance between the start of each row/column,
    // depending on the memory structure
    int a_stride = static_cast<int>(a.state == NORMAL ? a.n_rows+a.stride : a.n_columns+a.stride);
    int b_stride = static_cast<int>(b.state == NORMAL ? b.n_rows+b.stride : b.n_columns+b.stride);
    int out_stride = static_cast<int>(out.n_rows+out.stride);

	cblas_dgemm(CblasColMajor, a_state, b_state, M, N, K, scale, a.get_data(),
	      a_stride, b.get_data(), b_stride, 1.0, out.get_data(), out_stride);
}


void ActivationFunction::apply(Matrix in, Matrix out) const {
    transform(in.begin(), in.end(), out.begin(), *f);
}

void ActivationFunction::apply_deriv(Matrix in, Matrix d, Matrix out) const {
    transform(in.begin(), in.end(), out.begin(), *deriv);
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

void WinoutActivation::apply(Matrix in, Matrix out) const {
    for (size_t slice = 0; slice < in.n_slices; ++slice ) {
        for (size_t column = 0; column < in.n_columns; ++column ) {
            const int group_size = 2;
            for (size_t row = 0; row < in.n_rows; row += group_size) {
                d_type max = -1e10;
                size_t max_i = row;
                for (size_t row_g = row; row_g < row + group_size; row_g++) {
                    d_type current = in.get(row_g, column, slice);
                    if (current > max) {
                        max = current;
                        max_i = row_g;
                    }
                }
                for (size_t row_g = row; row_g < row + group_size; row_g++) {
                    if (row_g == max_i) {
                        out.get(row_g, column, slice) = in.get(row_g, column, slice);
                    }
                    else {
                        out.get(row_g, column, slice) = 0;
                    }
                }
            }
        }
    }
}

void WinoutActivation::apply_deriv(Matrix in, Matrix d, Matrix out) const {
    for (size_t pos = 0; pos < in.size; ++pos ) {
        if (in[pos] == 0) {
            out[pos] = 0;
        }
        else {
            out[pos] = d[pos];
        }
    }
}

void apply(Matrix in, Matrix out, unary_double_func f) {
	transform(in.begin(), in.end(), out.begin(), *f);
}

///Apply sigmoid to all units
void apply_sigmoid(Matrix a, Matrix out) {
    transform(a.begin(), a.end(), out.begin(), sigmoid);
}

void apply_sigmoid_deriv(Matrix a, Matrix out) {
    transform(a.begin(), a.end(), out.begin(), sigmoid_deriv);
}

///Apply tanh to all units
void apply_tanh(Matrix a, Matrix out) {
    transform(a.begin(), a.end(), out.begin(), tanh_);
}

void apply_tanh_deriv(Matrix a, Matrix out) {
    transform(a.begin(), a.end(), out.begin(), tanh_deriv);
}

///Apply tanh * 2to all units
void apply_tanhx2(Matrix a, Matrix out) {
    transform(a.begin(), a.end(), out.begin(), tanhx2);
}

void apply_tanhx2_deriv(Matrix a, Matrix out) {
    transform(a.begin(), a.end(), out.begin(), tanhx2_deriv);
}

///Copy the data of one matrix into another
void copy(Matrix a, Matrix b) {
    ASSERT(a.size <= b.size);
    if (a.stride == 0 && b.stride == 0 && a.state == b.state) {
        cblas_dcopy(static_cast<int>(a.size), a.get_data(), 1, b.get_data(), 1);
    } else {
        for (auto ita = a.begin(), itb = b.begin(); ita != a.end(); ++ita, ++itb) {
            *itb = *ita;
        }
    }
}

void squash(Matrix a, Matrix out) {
    out.set_all_elements_to(0.0);
    auto ito_end = out.end();
    for (auto ita = a.begin(), ito = out.begin(); ita != a.end(); ++ita, ++ito) {
        if (ito == ito_end)
            ito = out.begin();
        *ito += *ita;
    }
}

void squash(Matrix a, Matrix out, d_type scale) {
    out.set_all_elements_to(0.0);
    auto ito_end = out.end();
    for (auto ita = a.begin(), ito = out.begin(); ita != a.end(); ++ita, ++ito) {
        if (ito == ito_end)
            ito = out.begin();
        *ito += (*ita) * scale;
    }
}


///Elementwise multiplication, with squash to size of out (out is smaller than a and b)
void dot_squash(Matrix a, Matrix b, Matrix out) {
    ASSERT(a.size == b.size);
    ASSERT(a.size % out.size == 0);
    ASSERT(a.state == b.state && b.state == out.state);

    out.set_all_elements_to(0.0);
    auto ito_end = out.end();
    for (auto ita = a.begin(), itb = b.begin(), ito = out.begin(); ita != a.end(); ++ita, ++itb, ++ito) {
        if (ito == ito_end) {
            ito = out.begin();
        }
        *ito += *ita * *itb;
    }
}

///Elementwise multiplication, with squash to size of out (out is smaller than a and b)
void dot_squash(Matrix a, Matrix b, Matrix out, d_type scale) {
    ASSERT(a.size == b.size);
    ASSERT(a.size % out.size == 0);
    ASSERT(a.state == b.state && b.state == out.state);

    out.set_all_elements_to(0.0);
    auto ito_end = out.end();
    for (auto ita = a.begin(), itb = b.begin(), ito = out.begin(); ita != a.end(); ++ita, ++itb, ++ito) {
        if (ito == ito_end) {
            ito = out.begin();
        }
        *ito += (*ita) * (*itb) * scale;
    }
}


///scale matrix by a scalar
void scale_into(Matrix a, d_type alpha) {
    if (a.stride == 0) {
        cblas_dscal(static_cast<int>(a.size), alpha, a.get_data(), 1);
    } else {
        for (d_type& v : a) {
            v *= alpha;
        }
    }
}



///Elementwise add
void add(Matrix a, Matrix b, Matrix out) {
    ASSERT(a.size == b.size);
    ASSERT(a.size == out.size);
    Matrix::iterator ita = a.begin();
    Matrix::iterator ita_end = a.end();
    Matrix::iterator itb = b.begin();
    Matrix::iterator itout = out.begin();
    for (; ita != ita_end; ++ita, ++itb, ++itout) {
        *itout = *ita + *itb;
    }
}



void clip_elements(Matrix a, const d_type min, const d_type max) {
    for (auto it = a.begin(); it != a.end(); ++it) {
	    if (*it < min) {
	        *it = min;
	    } else if (*it > max) {
	        *it = max;
	    }
	}
}








/*




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
