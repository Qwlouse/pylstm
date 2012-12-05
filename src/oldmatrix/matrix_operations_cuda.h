/*
 * matrix_operations.h
 *
 *  Created on: Jun 14, 2011
 *      Author: stollenga
 */

#ifndef MATRIX_OPERATIONS_CUDA_H_
#define MATRIX_OPERATIONS_CUDA_H_

#include "matrix_cuda.h"

#include <cublas.h>

typedef double element_type;

inline void init_operations() {
	cublasInit();
}

inline void mask(MatrixPtrGPU arg1, MatrixPtrGPU mask) {
	size_t n_rows(arg1.rows());

	//to implement
}

inline void multiply(MatrixPtrGPU arg1, MatrixPtrGPU arg2, MatrixPtrGPU result) {
	assert(arg1.columns() == arg2.rows());
	assert(arg1.rows() == result.rows());
	assert(arg2.columns() == result.columns());

	cublasDgemm ('N', 'N' ,arg1.rows(), arg2.columns(), arg1.columns(),
		                           1.0, arg1.ptr(), arg1.rows(),
		                            arg2.ptr(), arg2.rows(), 1.0, result.ptr(),
		                            result.rows());
}

inline void multiply_transpose(MatrixPtrGPU arg1, MatrixPtrGPU arg2, MatrixPtrGPU result, element_type const alpha = 1.0, element_type const beta = 1.0) {
	assert(arg1.rows() == arg2.rows());
	assert(arg1.columns() == result.rows());
	assert(arg2.columns() == result.columns());

	cublasDgemm('T', 'N', arg1.columns(), arg2.columns(), arg1.rows(),
			alpha, arg1.ptr(), arg1.rows(), arg2.ptr(), arg2.rows(), beta, result.ptr(), result.rows());
}

inline void multiply_transpose_transpose(MatrixPtrGPU arg1, MatrixPtrGPU arg2, MatrixPtrGPU result, element_type const alpha = 1.0, element_type const beta = 1.0) {
	assert(arg1.rows() == arg2.columns());
	assert(arg1.columns() == result.rows());
	assert(arg2.rows() == result.columns());

	cublasDgemm('T', 'T', arg1.columns(), arg2.rows(), arg1.rows(),
			alpha, arg1.ptr(), arg1.rows(), arg2.ptr(), arg2.rows(), beta, result.ptr(), result.rows());
}

inline void multiply_normal_transpose(MatrixPtrGPU arg1, MatrixPtrGPU arg2, MatrixPtrGPU result, element_type const alpha = 1.0, element_type const beta = 1.0) {
	assert(arg1.columns() == arg2.columns());
	assert(arg1.rows() == result.rows());
	assert(arg2.rows() == result.columns());

	cublasDgemm('N', 'T', arg1.rows(), arg2.rows(), arg1.columns(),
			alpha, arg1.ptr(), arg1.rows(), arg2.ptr(), arg2.rows(), beta, result.ptr(), result.rows());
}

inline void multiply_normal_transpose_shifted(MatrixPtrGPU arg1, MatrixPtrGPU arg2, MatrixPtrGPU result, size_t n_batches, element_type const alpha = 1.0, element_type const beta = 1.0) {
	assert(arg1.columns() == arg2.columns());
	assert(arg1.rows() == result.rows());
	assert(arg2.rows() == result.columns());

	cublasDgemm('N', 'T', arg1.rows(), arg2.rows(), arg1.columns() - n_batches,
			alpha, arg1.ptr() + n_batches * arg2.rows(), arg1.rows(), arg2.ptr(), arg2.rows(), beta, result.ptr(), result.rows());
}

inline void multiply_normal_shifted(MatrixPtrGPU arg1, MatrixPtrGPU arg2, MatrixPtrGPU result, size_t n_batches, element_type const alpha = 1.0, element_type const beta = 1.0) {
	assert(arg1.columns() == arg2.columns());
	assert(arg1.rows() == result.rows());
	assert(arg2.rows() == result.columns());

	cublasDgemm('N', 'N', arg1.rows(), arg2.rows(), arg1.columns() - n_batches,
			alpha, arg1.ptr() + n_batches * arg2.rows(), arg1.rows(), arg2.ptr(), arg2.rows(), beta, result.ptr(), result.rows());
}

inline element_type norm2(MatrixPtrGPU arg1)
{
	return cublasDnrm2(arg1.size(), arg1.ptr(), 1);
}


//inline void multiply_vector(MatrixPtrGPU arg1, MatrixPtrGPU arg2, MatrixPtrGPU result, element_type const alpha = 1.0) {
//	cublasDgbmv('N', arg1.size(), arg1.size(), 0, 0, alpha,
//				arg1.ptr(), 1,
//				arg2.ptr(), 1,
//				0.0,
//				result.ptr(), 1);
//}

inline MatrixPtrGPU scale(MatrixPtrGPU arg1, element_type alpha) {
	cublasDscal(arg1.size(), alpha, arg1.ptr(), 1);
	return arg1;
}

inline void add(MatrixPtrGPU arg1, MatrixPtrGPU arg2) {
	cublasDaxpy(arg1.size(), 1.0, arg1.ptr(), 1, arg2.ptr(), 1);
}

inline void add(MatrixPtrGPU arg1, element_type arg2) {
	MatrixGPU scale(arg1.columns(), 1);
	scale.set(arg2);

	cublasDaxpy(arg1.size(), 1.0, scale.ptr(), 0, arg1.ptr(), 1);
}

//inline void square(MatrixPtrGPU arg1, MatrixPtrGPU result, element_type const alpha = 1.0) {
//	multiply_vector(arg1, arg1, result, alpha);
//}

inline void copy(MatrixPtrGPU const arg1, MatrixPtrGPU arg2) {
	assert(arg1.size() == arg2.size());
	cublasDcopy(arg1.size(), arg1.ptr(), 1, arg2.ptr(), 1);
}

inline void copy(MatrixPtrGPU const arg1, MatrixPtrGPU arg2, MatrixPtrGPU::size_type size) {
	cublasDcopy(size, arg1.ptr(), 1, arg2.ptr(), 1);
}

inline void copy(MatrixPtrGPU const arg1, size_t offset1, MatrixPtrGPU arg2, size_t offset2, MatrixPtrGPU::size_type size) {
	cublasDcopy(size, arg1.ptr() + offset1, 1, arg2.ptr() + offset2, 1);
}

inline void copy(MatrixPtr const arg1, MatrixPtrGPU arg2) {
	assert(arg1.size() == arg2.size());
	cublasSetVector(arg1.size(), sizeof(element_type), arg1.ptr(), 1, arg2.ptr(), 1);
}

inline void copy(MatrixPtr const arg1, size_t offset1, MatrixPtrGPU arg2, size_t offset2, size_t size) {
	cublasSetVector(size, sizeof(element_type), arg1.ptr() + offset1, 1, arg2.ptr() + offset2, 1);
}

inline void copy(MatrixPtrGPU const arg1, MatrixPtr arg2) {
	assert(arg1.size() == arg2.size());
	cublasGetVector(arg2.size(), sizeof(element_type), arg1.ptr(), 1, arg2.ptr(), 1);
}

//inline void copy(MatrixPtrGPU arg1, std::vector<element_type>::iterator &arg2) {
//	Matrix intermediate(arg1.size(), 1);
//	copy(arg1, intermediate);
//	std::copy(intermediate.begin(), intermediate.end(), arg2);
//}

//inline void copy(MatrixPtrGPU const arg1, std::vector<element_type>::iterator arg2);
//{
//	cublasGetVector(arg1.size(), sizeof(element_type), arg1.ptr(), 1, &*arg2, 1);
//}

inline void diff(MatrixPtrGPU arg1, MatrixPtrGPU arg2) {
	cublasDaxpy(arg1.size(), -1.0, arg1.ptr(), 1, arg2.ptr(), 1);
	cublasDscal(arg2.size(), -1.0, arg2.ptr(), 1);
}



//inline void apply_function(MatrixGPU &arg, element_type_function func) {
//	MatrixGPU new_mat(arg.rows(), arg.columns());
//	thrust::transform(thrust::device_ptr<element_type>(arg.ptr()), thrust::device_ptr<element_type>(arg.ptr() + 1), thrust::device_ptr<element_type>(new_mat.begin()), tanh);
//}
//
//
//void apply_function(MatrixPtrGPU arg, element_type_function func) {
//	MatrixPtrGPU::iterator it(arg.begin());
//	MatrixPtrGPU::iterator end(arg.end());
//
//	for (; it != end; ++it)
//		*it = (*func)(*it);
//}

inline void squash(MatrixPtrGPU arg1, MatrixPtrGPU result, element_type alpha = 1.0) {
	assert(result.size() == arg1.rows());

	MatrixGPU scale(arg1.columns(), 1);
	scale.set(1);

	cublasDgemv('N', arg1.rows(), arg1.columns(), alpha, arg1.ptr(), arg1.rows(), scale.begin(), 1, 1.0, result.ptr(), 1);
}



//void fill_random(Matrix *arg) {
//	Matrix &mat(*arg);
//	Matrix::iterator it(mat.begin()), end(mat.end());
//
//	for (; it != end; ++it)
//		*it = rand() % 4;
//}

#endif /* MATRIX_OPERATIONS_H_ */
