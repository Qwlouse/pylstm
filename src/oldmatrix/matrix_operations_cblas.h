/*
 * matrix_operations.h
 *
 *  Created on: Jun 14, 2011
 *      Author: stollenga
 */

#ifndef MATRIX_OPERATIONS_CBLAS_H_
#define MATRIX_OPERATIONS_CBLAS_H_

#include "matrix.h"

#include <gsl/gsl_cblas.h>
#include <algorithm>
#include <cmath>

typedef element_type (*element_type_function)(element_type val);

inline void multiply(MatrixPtr arg1, MatrixPtr arg2, MatrixPtr result) {
	assert(arg1.columns() == arg2.rows());
	assert(arg1.rows() == result.rows());
	assert(arg2.columns() == result.columns());

	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, arg1.rows(), arg2.columns(), arg1.columns(),
			1.0, arg1.ptr(), arg1.rows(), arg2.ptr(), arg2.rows(), 1.0, result.ptr(), result.rows());
}

inline void multiply_transpose(MatrixPtr arg1, MatrixPtr arg2, MatrixPtr result, element_type const alpha = 1.0, element_type const beta = 1.0) {
	assert(arg1.rows() == arg2.rows());
	assert(arg1.columns() == result.rows());
	assert(arg2.columns() == result.columns());

	cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, arg1.columns(), arg2.columns(), arg1.rows(),
			alpha, arg1.ptr(), arg1.rows(), arg2.ptr(), arg2.rows(), beta, result.ptr(), result.rows());
}

inline void multiply_transpose_transpose(MatrixPtr arg1, MatrixPtr arg2, MatrixPtr result, element_type const alpha = 1.0, element_type const beta = 1.0) {
	assert(arg1.rows() == arg2.columns());
	assert(arg1.columns() == result.rows());
	assert(arg2.rows() == result.columns());

	cblas_sgemm(CblasColMajor, CblasTrans, CblasTrans, arg1.columns(), arg2.rows(), arg1.rows(),
			alpha, arg1.ptr(), arg1.rows(), arg2.ptr(), arg2.rows(), beta, result.ptr(), result.rows());
}

inline void multiply_normal_transpose(MatrixPtr arg1, MatrixPtr arg2, MatrixPtr result, element_type const alpha = 1.0, element_type const beta = 1.0) {
	assert(arg1.columns() == arg2.columns());
	assert(arg1.rows() == result.rows());
	assert(arg2.rows() == result.columns());

	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, arg1.rows(), arg2.rows(), arg1.columns(),
			alpha, arg1.ptr(), arg1.rows(), arg2.ptr(), arg2.rows(), beta, result.ptr(), result.rows());
}

inline void multiply_normal_transpose_shifted(MatrixPtr arg1, MatrixPtr arg2, MatrixPtr result, size_t n_batches, element_type const alpha = 1.0, element_type const beta = 1.0) {
	assert(arg1.columns() == arg2.columns());
	assert(arg1.rows() == result.rows());
	assert(arg2.rows() == result.columns());

	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, arg1.rows(), arg2.rows(), arg1.columns() - n_batches,
			alpha, arg1.ptr() + n_batches * arg2.rows(), arg1.rows(), arg2.ptr(), arg2.rows(), beta, result.ptr(), result.rows());
}

inline void multiply_normal_shifted(MatrixPtr arg1, MatrixPtr arg2, MatrixPtr result, size_t n_batches, element_type const alpha = 1.0, element_type const beta = 1.0) {
	assert(arg1.columns() == arg2.columns());
	assert(arg1.rows() == result.rows());
	assert(arg2.rows() == result.columns());

	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, arg1.rows(), arg2.rows(), arg1.columns() - n_batches,
			alpha, arg1.ptr() + n_batches * arg2.rows(), arg1.rows(), arg2.ptr(), arg2.rows(), beta, result.ptr(), result.rows());
}

inline element_type norm2(MatrixPtr arg1)
{
	return cblas_snrm2(arg1.size(), arg1.ptr(), 1);
}

inline element_type sum(MatrixPtr arg1)
{
	return cblas_sasum(arg1.size(), arg1.ptr(), 1);
}

inline void multiply_vector(MatrixPtr arg1, MatrixPtr arg2, MatrixPtr result, element_type beta = 0.0) {
	if (arg2.size() < arg1.size()) {
		Matrix new_arg2(arg1.rows(), arg1.columns());
		for (size_t i(0); i <= arg1.size() - arg2.size(); i += arg2.size())
			cblas_scopy(arg2.size(), arg2.ptr(), 1, new_arg2.ptr() + i, 1);

		cblas_sgbmv(CblasColMajor, CblasNoTrans, arg1.size(), arg1.size(), 0, 0, 1.0,
					arg1.ptr(), 1,
			                new_arg2.ptr(), 1,
					beta,
					result.ptr(), 1);
	}
	else
		cblas_sgbmv(CblasColMajor, CblasNoTrans, arg1.size(), arg1.size(), 0, 0, 1.0,
					arg1.ptr(), 1,
					arg2.ptr(), 1,
					beta,
					result.ptr(), 1);

}

inline void multiply_vector_shifted(MatrixPtr arg1, MatrixPtr arg2, MatrixPtr result, size_t shift, element_type beta = 0.0) {
	MatrixPtr::size_type len(arg1.size() - shift * arg1.rows());

	if (arg2.size() < arg1.size()) {
		Matrix new_arg2(arg1.rows(), arg1.columns());
		for (size_t i(0); i <= arg1.size() - arg2.size(); i += arg2.size())
			cblas_scopy(arg2.size(), arg2.ptr(), 1, new_arg2.ptr() + i, 1);

		cblas_sgbmv(CblasColMajor, CblasNoTrans, len, len, 0, 0, 1.0,
					arg1.ptr() + arg1.rows() * shift, 1,
					new_arg2.ptr(), 1,
					beta,
					result.ptr(), 1);
	}
	else
		cblas_sgbmv(CblasColMajor, CblasNoTrans, len, len, 0, 0, 1.0,
				arg1.ptr() + arg1.rows() * shift, 1,
				arg2.ptr(), 1,
				beta,
				result.ptr(), 1);
}
//
//inline void multiply_vector_shifted(MatrixPtr arg1, MatrixPtr arg2, MatrixPtr result, size_t shift, element_type beta = 0.0) {
//	MatrixPtr::size_type len(arg1.size() - shift * arg1.rows());
//	dgbmv(&NO_TRANS, &len, &len, &diff_zero, &diff_zero, &element_type_one,
//				arg1.ptr() + arg1.rows() * shift, &diff_one,
//				arg2.ptr(), &diff_one,
//				&beta,
//				result.ptr(), &diff_one);
//}

inline MatrixPtr scale(MatrixPtr arg1, element_type alpha) {
	cblas_sscal(arg1.size(), alpha, arg1.ptr(), 1);
	return arg1;
}

inline void add(MatrixPtr arg1, MatrixPtr arg2) {
	cblas_saxpy(arg1.size(), 1.0, arg1.ptr(), 1, arg2.ptr(), 1);
}

inline void add(MatrixPtr arg1, element_type arg2) {
	MatrixPtr::iterator it(arg1.begin());
	MatrixPtr::iterator end(arg1.end());
	for (;it != end; ++it) *it += arg2;
}

inline void square(MatrixPtr arg1, MatrixPtr result, element_type const alpha = 1.0) {
	cblas_sgbmv(CblasColMajor, CblasNoTrans, arg1.size(), arg1.size(), 0, 0, alpha,
				arg1.ptr(), 1,
				arg1.ptr(), 1,
				0.0,
				result.ptr(), 1);
}

inline void copy(MatrixPtr const arg1, MatrixPtr arg2) {
	cblas_scopy(arg1.size(), arg1.ptr(), 1, arg2.ptr(), 1);
}

inline void copy(MatrixPtr const arg1, MatrixPtr arg2, MatrixPtr::size_type size) {
	cblas_scopy(size, arg1.ptr(), 1, arg2.ptr(), 1);
}

inline void copy(MatrixPtr arg1, std::vector<element_type>::iterator arg2) {
	std::copy(arg1.begin(), arg1.end(), arg2);
}


inline void diff(MatrixPtr arg1, MatrixPtr arg2) {
	cblas_saxpy(arg1.size(), -1.0, arg1.ptr(), 1, arg2.ptr(), 1);
	cblas_sscal(arg2.size(), -1.0, arg2.ptr(), 1);
}

inline void add_vector(MatrixPtr arg1, MatrixPtr arg2) {
//	   const enum CBLAS_ORDER Order,
//	   const enum CBLAS_TRANSPOSE TransA,
//	   const enum CBLAS_TRANSPOSE TransB,
//	   const int M,
//	   const int N,
//	   const int K,
//	   const element_type alpha,
//	   const element_type *A,
//	   const int lda,
//	   const element_type *B,
//	   const int ldb,
//	   const element_type beta,
//	   element_type *C,
//	   const int ldc
//	CblasColMajor, CblasNoTrans, CblasTrans, arg1.rows(), arg2.rows(), arg1.columns() - n_batches,
//				alpha, arg1.ptr() + n_batches * arg2.rows(), arg1.rows(), arg2.ptr(), arg2.rows(), beta, result.ptr(), result.rows()
//	cblas_sgemm (CblasColMajor, CblasNoTrans, CblasNoTrans,
//
//			);

	MatrixPtr::iterator it(arg1.begin()), end(arg1.end());
	MatrixPtr::iterator it2(arg2.begin()), end2(arg2.end());

	for (; it != end; ++it, ++it2) {
		if (it2 == end2)
			it2 = arg2.begin();
		*it += *it2;
	}
}

//inline void apply_function(Matrix &arg, element_type_function func) {
//	Matrix::iterator it(arg.begin());
//	Matrix::iterator end(arg.end());
//
//	for (; it != end; ++it)
//		*it = (*func)(*it);
//}


inline void apply_function(MatrixPtr arg, element_type_function func) {
	MatrixPtr::iterator it(arg.begin());
	MatrixPtr::iterator end(arg.end());

	for (; it != end; ++it)
		*it = (*func)(*it);
}

inline void apply_softmax(MatrixPtr arg1, MatrixPtr arg2) {
  std::vector<float> totals(arg1.columns());

  MatrixPtr::iterator it(arg1.begin());
  MatrixPtr::iterator end(arg1.end());
  MatrixPtr::iterator it2(arg2.begin());

  std::vector<float>::iterator total_it = totals.begin();
  size_t counter(0), count_to(arg1.rows());
  for (; it != end; ++it, ++it2, ++counter) {
	  if (counter == count_to) {counter = 0; ++total_it;}
    *it2 = exp(*it); //exp(*it / temperature);
    *total_it += *it2;
  }

  total_it = totals.begin();
  counter = 0;
  {
    MatrixPtr::iterator it(arg2.begin());
    MatrixPtr::iterator end(arg2.end());
    for (; it != end; ++it, ++counter) {
  	  if (counter == count_to) {counter = 0; ++total_it;}
      *it = *it / *total_it; //exp(*it / temperature);
    }
  }
}

inline void softmax_deriv(MatrixPtr in_deltas, MatrixPtr activations, MatrixPtr deltas) {
  std::vector<float> totals(activations.columns());

  MatrixPtr::iterator it(activations.begin());
  MatrixPtr::iterator end(activations.end());

  std::vector<float>::iterator total_it = totals.begin();
  size_t counter(0), count_to(activations.rows());
  for (; it != end; ++it, ++counter) {
    if (counter == count_to) {counter = 0; ++total_it;}
    *total_it += exp(*it);
  }

  total_it = totals.begin();

  it = activations.begin();
  MatrixPtr::iterator delta_it(deltas.begin());
  MatrixPtr::iterator in_delta_it(in_deltas.begin());
  for (; it != end; ++total_it, it += count_to, in_delta_it += count_to) {
    MatrixPtr::iterator ref(it);
    MatrixPtr::iterator row_it(ref), row_end(ref + count_to);
    
    for(; row_it != row_end; ++row_it, ++delta_it) {
      MatrixPtr::iterator row2_it(ref), row2_end(ref + count_to);
      MatrixPtr::iterator in_delta_loop(in_delta_it);
      for(; row2_it != row2_end; ++row2_it, ++in_delta_loop) {
	if (row_it == row2_it)
	  *delta_it += *in_delta_loop * (*row_it * *total_it - *row_it * *row_it) / (*total_it * *total_it);
	else
	  *delta_it += *in_delta_loop * (-*row_it) * *row2_it / (*total_it * *total_it);
      }
    }
  }
}

inline void apply_softmax(MatrixPtr arg) {
  apply_softmax(arg, arg);
}


inline void apply_tanh(MatrixPtr arg1, MatrixPtr arg2) {
	MatrixPtr::iterator it(arg1.begin());
	MatrixPtr::iterator end(arg1.end());
	MatrixPtr::iterator it2(arg2.begin());
	for (; it != end; ++it, ++it2)
		*it2 = 1.0 * tanh(*it);
}

inline void apply_tanh(MatrixPtr arg) {
	apply_tanh(arg, arg);
}

inline void apply_sigmoid(MatrixPtr arg1, MatrixPtr arg2) {
	MatrixPtr::iterator it(arg1.begin());
	MatrixPtr::iterator end(arg1.end());
	MatrixPtr::iterator it2(arg2.begin());
	for (; it != end; ++it, ++it2)
		*it2 = 1.0 / (1.0 + exp(-*it));
}

inline void apply_sigmoid(MatrixPtr arg) {
	apply_sigmoid(arg, arg);
}


inline void apply_log(MatrixPtr arg1, MatrixPtr arg2) {
	MatrixPtr::iterator it(arg1.begin());
	MatrixPtr::iterator end(arg1.end());
	MatrixPtr::iterator it2(arg2.begin());
	for (; it != end; ++it, ++it2)
		*it2 = log(*it);
}

inline void apply_log(MatrixPtr arg) {
	apply_log(arg, arg);
}


inline void squash(MatrixPtr arg1, MatrixPtr result, element_type alpha = 1.0) {
	assert(result.size() == arg1.rows());

	std::vector<element_type> scale(arg1.columns(), 1);

	cblas_sgemv(CblasColMajor, CblasNoTrans, arg1.rows(), arg1.columns(), alpha, arg1.ptr(), arg1.rows(), &scale[0], 1, 1.0, result.ptr(), 1);
}


#endif /* MATRIX_OPERATIONS_H_ */
