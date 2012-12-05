/*
 * matrix_operations.h
 *
 *  Created on: Jun 14, 2011
 *      Author: stollenga
 */

#ifndef MATRIX_OPERATIONS_MATLABBLAS_H_
#define MATRIX_OPERATIONS_MATLABBLAS_H_

#include "matrix.h"
#include "except.h"

#include <blas.h>
#include <algorithm>
#include <numeric>
#include <math.h>

typedef element_type (*double_function)(element_type val);
static element_type double_one = 1.0;
static element_type double_zero = 0.0;
static element_type double_min_one = -1.0;
static char NO_TRANS = 'N';
static char TRANS = 'T';
static ptrdiff_t diff_one = 1;
static ptrdiff_t diff_zero = 0;


inline void multiply(MatrixPtr arg1, MatrixPtr arg2, MatrixPtr result) {
	assert(arg1.columns() == arg2.rows());
	assert(arg1.rows() == result.rows());
	assert(arg2.columns() == result.columns());

	dgemm(&NO_TRANS, &NO_TRANS, &arg1.rows(), &arg2.columns(), &arg1.columns(),
			&double_one,
			arg1.ptr(),
			&arg1.rows(), arg2.ptr(), &arg2.rows(), &double_one, result.ptr(), &result.rows());
}

inline void multiply_transpose(MatrixPtr arg1, MatrixPtr arg2, MatrixPtr result, element_type alpha = 1.0, element_type beta = 1.0) {
	assert(arg1.rows() == arg2.rows());
	assert(arg1.columns() == result.rows());
	assert(arg2.columns() == result.columns());

	dgemm(&TRANS, &NO_TRANS, &arg1.columns(), &arg2.columns(), &arg1.rows(),
			&alpha, arg1.ptr(), &arg1.rows(), arg2.ptr(), &arg2.rows(),
			&beta, result.ptr(),
			&result.rows());
}

inline void multiply_transpose_transpose(MatrixPtr arg1, MatrixPtr arg2, MatrixPtr result, element_type alpha = 1.0, element_type beta = 1.0) {
	assert(arg1.rows() == arg2.columns());
	assert(arg1.columns() == result.rows());
	assert(arg2.rows() == result.columns());

	dgemm(&TRANS, &TRANS, &arg1.columns(), &arg2.rows(), &arg1.rows(),
			&alpha, arg1.ptr(), &arg1.rows(), arg2.ptr(), &arg2.rows(), &beta, result.ptr(), &result.rows());
}

inline void multiply_normal_transpose(MatrixPtr arg1, MatrixPtr arg2, MatrixPtr result, element_type alpha = 1.0, element_type beta = 1.0) {
	assert(arg1.columns() == arg2.columns());
	assert(arg1.rows() == result.rows());
	assert(arg2.rows() == result.columns());

	dgemm(&NO_TRANS, &TRANS, &arg1.rows(), &arg2.rows(), &arg1.columns(),
			&alpha, arg1.ptr(), &arg1.rows(), arg2.ptr(), &arg2.rows(), &beta, result.ptr(), &result.rows());
}

inline void multiply_normal_transpose_shifted(MatrixPtr arg1, MatrixPtr arg2, MatrixPtr result, size_t n_batches, element_type alpha = 1.0, element_type beta = 1.0) {
	assert(arg1.columns() == arg2.columns());
	assert(arg1.rows() == result.rows());
	assert(arg2.rows() == result.columns());
	MatrixPtr::size_type len = arg1.columns() - n_batches;
	dgemm(&NO_TRANS, &TRANS, &arg1.rows(), &arg2.rows(), &len,
			&alpha, arg1.ptr() + n_batches * arg2.rows(), &arg1.rows(), arg2.ptr(), &arg2.rows(), &beta, result.ptr(), &result.rows());
}

inline void multiply_normal_shifted(MatrixPtr arg1, MatrixPtr arg2, MatrixPtr result, size_t n_batches, element_type alpha = 1.0, element_type beta = 1.0) {
	assert(arg1.columns() == arg2.columns());
	assert(arg1.rows() == result.rows());
	assert(arg2.rows() == result.columns());
	MatrixPtr::size_type len = arg1.columns() - n_batches;
	dgemm(&NO_TRANS, &NO_TRANS, &arg1.rows(), &arg2.rows(), &len,
			&alpha, arg1.ptr() + n_batches * arg2.rows(), &arg1.rows(), arg2.ptr(), &arg2.rows(), &beta, result.ptr(), &result.rows());
}

inline element_type norm2(MatrixPtr arg1)
{
	MatrixPtr::size_type len(arg1.size());
	return dnrm2(&len, arg1.ptr(), &diff_one);
}

inline element_type sum(MatrixPtr arg1)
{
  return std::accumulate(arg1.begin(), arg1.end(), 0.0);
  //MatrixPtr::size_type len(arg1.size());
  //return dasum(&len, arg1.ptr(), &diff_one);
}

inline void multiply_vector(MatrixPtr arg1, MatrixPtr arg2, MatrixPtr result, element_type beta = 0.0) {
	MatrixPtr::size_type len(arg1.size());


	if (arg2.size() < arg1.size()) {
		Matrix new_arg2(arg1.rows(), arg1.columns());
		MatrixPtr::size_type arg2_len(arg2.size());

		for (MatrixPtr::size_type i(0); i <= arg1.size() - arg2.size(); i += arg2.size())
		  dcopy(&arg2_len, arg2.ptr(), &diff_one, new_arg2.ptr() + i, &diff_one);

		dgbmv(&NO_TRANS, &len, &len, &diff_zero, &diff_zero, &double_one,
					arg1.ptr(), &diff_one,
					new_arg2.ptr(), &diff_one,
					&beta,
					result.ptr(), &diff_one);
	}
	else
	  dgbmv(&NO_TRANS, &len, &len, &diff_zero, &diff_zero, &double_one,
					arg1.ptr(), &diff_one,
					arg2.ptr(), &diff_one,
					&beta,
					result.ptr(), &diff_one);
}

inline void multiply_vector_shifted(MatrixPtr arg1, MatrixPtr arg2, MatrixPtr result, size_t shift, element_type beta = 0.0) {
	MatrixPtr::size_type len(arg1.size() - shift * arg1.rows());
	dgbmv(&NO_TRANS, &len, &len, &diff_zero, &diff_zero, &double_one,
				arg1.ptr() + arg1.rows() * shift, &diff_one,
				arg2.ptr(), &diff_one,
				&beta,
				result.ptr(), &diff_one);

	if (arg2.size() < arg1.size()) {
		Matrix new_arg2(arg1.rows(), arg1.columns());
		MatrixPtr::size_type arg2_len(arg2.size());
		for (MatrixPtr::size_type i(0); i <= arg1.size() - arg2.size(); i += arg2.size())
		  dcopy(&arg2_len, arg2.ptr(), &diff_one, new_arg2.ptr() + i, &diff_one);

		dgbmv(&NO_TRANS, &len, &len, &diff_zero, &diff_zero, &double_one,
					arg1.ptr() + arg1.rows() * shift, &diff_one,
					new_arg2.ptr(), &diff_one,
					&beta,
					result.ptr(), &diff_one);
	}
	else
	  dgbmv(&NO_TRANS, &len, &len, &diff_zero, &diff_zero, &double_one,
					arg1.ptr() + arg1.rows() * shift, &diff_one,
					arg2.ptr(), &diff_one,
					&beta,
					result.ptr(), &diff_one);
}

inline MatrixPtr scale(MatrixPtr arg1, element_type alpha) {
	MatrixPtr::size_type len(arg1.size());
	dscal(&len, &alpha, arg1.ptr(), &diff_one);
	return arg1;
}

inline void add(MatrixPtr arg1, MatrixPtr arg2) {
	MatrixPtr::size_type len(arg1.size());
	daxpy(&len, &double_one, arg1.ptr(), &diff_one, arg2.ptr(), &diff_one);
}

inline void add(MatrixPtr arg1, element_type arg2) {
	MatrixPtr::iterator it(arg1.begin());
	MatrixPtr::iterator end(arg1.end());
	for (;it != end; ++it) *it += arg2;
}

inline void square(MatrixPtr arg1, MatrixPtr result, element_type alpha = 1.0) {
	MatrixPtr::size_type len(arg1.size());
	dgbmv(&NO_TRANS, &len, &len, &diff_zero, &diff_zero, &alpha,
				arg1.ptr(), &diff_one,
				arg1.ptr(), &diff_one,
				&double_zero,
				result.ptr(), &diff_one);
}

inline void copy(MatrixPtr arg1, MatrixPtr arg2) {
	MatrixPtr::size_type len(arg1.size());
	dcopy(&len, arg1.ptr(), &diff_one, arg2.ptr(), &diff_one);
}

inline void copy(MatrixPtr arg1, MatrixPtr arg2, MatrixPtr::size_type size) {
	MatrixPtr::size_type len(size);
	dcopy(&len, arg1.ptr(), &diff_one, arg2.ptr(), &diff_one);
}

inline void copy(MatrixPtr arg1, size_t offset1, MatrixPtr arg2, size_t offset2, MatrixPtr::size_type size) {
	MatrixPtr::size_type len(size);
	dcopy(&len, arg1.ptr() + offset1, &diff_one, arg2.ptr() + offset2, &diff_one);
}

inline void copy(MatrixPtr arg1, std::vector<element_type>::iterator &arg2) {
	std::copy(arg1.begin(), arg1.end(), arg2);
}


inline void diff(MatrixPtr arg1, MatrixPtr arg2) {
	MatrixPtr::size_type len1(arg1.size());
	MatrixPtr::size_type len2(arg2.size());
	daxpy(&len1, &double_min_one, arg1.ptr(), &diff_one, arg2.ptr(), &diff_one);
	dscal(&len2, &double_min_one, arg2.ptr(), &diff_one);
}

inline void mask(MatrixPtr arg1, MatrixPtr mask) {
	size_t n_rows(arg1.rows());

	MatrixPtr::ptr_type ptr_1(arg1.ptr()), ptr_2(mask.ptr()), end_2(mask.ptr() + mask.size());
	for (;ptr_2 < end_2; ++ptr_2)
		if (*ptr_2 == 0)
			for (size_t i(0); i < n_rows; ++i, ++ptr_1)
				*ptr_1 = 0.0;
		else
			ptr_1 += n_rows;
}

inline void add_vector(MatrixPtr arg1, MatrixPtr arg2) {
	MatrixPtr::iterator it(arg1.begin()), end(arg1.end());
	MatrixPtr::iterator it2(arg2.begin()), end2(arg2.end()), begin2(arg2.begin());

	for (; it != end; ++it, ++it2) {
		if (it2 == end2)
			it2 = begin2;
		*it += *it2;
	}
}

//inline void apply_function(Matrix &arg, double_function func) {
//	Matrix::iterator it(arg.begin());
//	Matrix::iterator end(arg.end());
//
//	for (; it != end; ++it)
//		*it = (*func)(*it);
//}


inline void apply_function(MatrixPtr arg, double_function func) {
	MatrixPtr::iterator it(arg.begin());
	MatrixPtr::iterator end(arg.end());

	for (; it != end; ++it)
		*it = (*func)(*it);
}

inline void apply_tanh(MatrixPtr arg1, MatrixPtr arg2) {
	MatrixPtr::iterator it(arg1.begin());
	MatrixPtr::iterator end(arg1.end());
	MatrixPtr::iterator it2(arg2.begin());
	for (; it != end; ++it, ++it2)
	  *it2 = 1.0 * tanh(*it);
}

inline void apply_tanh2(MatrixPtr arg1, MatrixPtr arg2) {
	MatrixPtr::iterator it(arg1.begin());
	MatrixPtr::iterator end(arg1.end());
	MatrixPtr::iterator it2(arg2.begin());
	for (; it != end; ++it, ++it2)
	  *it2 = 2.0 * tanh(*it);
}

inline void apply_tanh(MatrixPtr arg) {
	apply_tanh(arg, arg);
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


inline void apply_softmax(MatrixPtr arg1, MatrixPtr arg2) {
  std::vector<double> totals(arg1.columns());

  MatrixPtr::iterator it(arg1.begin());
  MatrixPtr::iterator end(arg1.end());
  MatrixPtr::iterator it2(arg2.begin());

  std::vector<double>::iterator total_it = totals.begin();
//  std::cout << arg1.rows() << std::endl;
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

inline void apply_max(MatrixPtr arg1) {
  MatrixPtr::iterator it(arg1.begin());
  MatrixPtr::iterator end(arg1.end());

  size_t counter(0), count_to(arg1.rows());
  double max_val(-99999999);
  MatrixPtr::iterator win(arg1.begin());
  for (; it != end; ++it, ++counter) {
    if (counter == count_to) {max_val = -999999999; counter = 0;}
    if (*it > max_val) {max_val = *it; win = it;}
    *it = 0.0;
    if (counter == count_to - 1) {*win = 1.0;}
  }
  //*win = 1.0;

}

inline void softmax_deriv(MatrixPtr in_deltas, MatrixPtr activations, MatrixPtr activation_tmp, MatrixPtr deltas) {
  std::vector<double> totals(activations.columns());

  MatrixPtr::iterator it(activations.begin());
  MatrixPtr::iterator end(activations.end());
  MatrixPtr::iterator it2(activation_tmp.begin());

  std::vector<double>::iterator total_it = totals.begin();
  size_t counter(0), count_to(activations.rows());
  for (; it != end; ++it, ++it2, ++counter) {
    if (counter == count_to) {counter = 0; ++total_it;}
    *it2 = exp(*it);
    *total_it += *it2;
  }

  total_it = totals.begin();

  it = activation_tmp.begin();
  end = activation_tmp.end();

  MatrixPtr::iterator delta_it(deltas.begin());
  MatrixPtr::iterator in_delta_it(in_deltas.begin());

  for (; it != end; ++total_it, it += count_to, in_delta_it += count_to) {
	 double total_2(*total_it * *total_it);
    MatrixPtr::iterator ref(it);
    MatrixPtr::iterator row_it(ref), row_end(ref + count_to);
    
    for(; row_it != row_end; ++row_it, ++delta_it) {
      MatrixPtr::iterator row2_it(ref), row2_end(ref + count_to);
      MatrixPtr::iterator in_delta_loop(in_delta_it);
      for(; row2_it != row2_end; ++row2_it, ++in_delta_loop) {
		if (row_it == row2_it)
		  *delta_it += *in_delta_loop * (*row_it * *total_it - *row_it * *row_it) / total_2;
		else
		  *delta_it += -*in_delta_loop * (*row_it * *row2_it) / total_2;
	//std::cout << *delta_it << " ";
      }
    }
  }  
}

inline void apply_softmax(MatrixPtr arg) {
  apply_softmax(arg, arg);
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

inline void squash(MatrixPtr arg1, MatrixPtr result, element_type alpha = 1.0) {
	assert(result.size() == arg1.rows());

	std::vector<element_type> scale(arg1.columns(), 1);

	dgemv(&NO_TRANS, &arg1.rows(), &arg1.columns(), &alpha, arg1.ptr(), &arg1.rows(), &scale[0], &diff_one, &double_one, result.ptr(), &diff_one);
}

//inline void fill_random(Matrix *arg) {
//	Matrix &mat(*arg);
//	Matrix::iterator it(mat.begin()), end(mat.end());
//
//	for (; it != end; ++it)
//		*it = rand() % 4;
//}

inline bool compare(MatrixPtr arg1, MatrixPtr arg2) {
  //  assert(arg1.size() == arg2.rows());
  if(arg1.size() != arg2.size()) return false;
  
  for(int i=0; i<arg1.size(); ++i) {
    if (arg1[i] != arg2[i]) return false;
  }
  
  return true;
}

inline int count_wrong_sequences(Matrix3D &arg1, Matrix3D &arg2) {
  //  assert(arg1.size() == arg2.rows());
  if(arg1.size() != arg2.size()) 
    throw Except("sizes dont match");
  
  size_t count(0);
  for(int b = 0; b < arg1.columns(); ++b) {
    bool different(false);
    for (size_t t(0); t < arg1.slices(); ++t) {
      for (size_t r(0); r < arg1.rows(); ++r) {
	size_t index(r + (b + t * arg1.columns()) * arg1.rows());
	if (arg1[index] != arg2[index]) {
	  count += 1;
	  different = true;
	  break;
	}
      }
      if (different)
	break;
    }
  }
  
  return count;
}


#endif /* MATRIX_OPERATIONS_H_ */
