/*
 * matrix.h
 *
 *  Created on: Jun 14, 2011
 *      Author: stollenga
 */

//#ifdef USE_GPU

#ifndef MATRIX_CUDA_H_
#define MATRIX_CUDA_H_

#include <vector>
#include <algorithm>
#include <iostream>
#include <cuda.h>
#include <cublas.h>
#include <cuda_runtime_api.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>

#include "matrix.h"
#include "defines.h"
#include "except.h"

class MatrixPtrGPU {
public:

	typedef long int size_type;

private:
	element_type* d_data_ptr;
	size_type d_rows;
	size_type d_columns;
	size_type d_size;

public:
	typedef element_type *iterator;

	inline MatrixPtrGPU(element_type *data_ptr, size_type rows, size_type columns) :
		d_data_ptr(data_ptr),
		d_rows(rows),
		d_columns(columns),
		d_size(d_rows * d_columns){
	}

	inline MatrixPtrGPU(size_type rows, size_type columns) :
		d_data_ptr(0),
		d_rows(rows),
		d_columns(columns),
		d_size(d_rows * d_columns){
	}

	inline MatrixPtrGPU(MatrixPtr const other) :
	  d_data_ptr(0),
	  d_rows(other.d_rows),
	  d_columns(other.d_columns),
	  d_size(d_rows * d_columns){
	}

	inline element_type *ptr() {	return d_data_ptr;}
	inline element_type const *ptr() const  {	return d_data_ptr;}

	void set_ptr(element_type *ptr) {
		d_data_ptr = ptr;
	}

	inline size_type size() const {
		return d_size;
	}
//	void set(std::vector<element_type> &v);

	inline size_type &columns() {
		return d_columns;
	}

	inline size_type &rows() {
		return d_rows;
	}


	inline element_type &operator[](size_type i) {
		return d_data_ptr[i];
	}

	inline element_type const &operator[](size_type i) const {
		return d_data_ptr[i];
	}

	inline element_type *column_start(size_type index) {
		return &d_data_ptr[index * d_rows];
	}

	inline element_type *slice_start(size_type index) {
		return &d_data_ptr[index * d_size];
	}
//	void print() ;

	void set_n(size_t n, element_type value);

	iterator begin();

	iterator end();

	void set(element_type value);

	void clear();

	void fill_random(element_type std);
	void fill_random(element_type std, element_type thresh) {throw Except("not implemented");}
	void fill_random_positive(element_type std);

	std::vector<element_type> to_vector();

};

class MatrixGPU {
public:
	typedef long int size_type;
	typedef MatrixPtrGPU ptr_type;
	typedef element_type* raw_ptr_type;

private:
	raw_ptr_type d_data;
	size_type d_rows;
	size_type d_columns;
	size_type d_size;

	thrust::device_ptr<element_type> d_thrust_pointer;

public:
	typedef element_type * iterator;

	inline MatrixGPU(size_type rows = 0, size_type columns = 0) :
		d_data(0),
		d_rows(rows),
		d_columns(columns),
		d_size(d_rows * d_columns)
	{
	  //std::cerr << "allocating: " << size() << std::endl;
	  if (size())
	    if (CUBLAS_STATUS_SUCCESS != cublasAlloc(size(), sizeof(element_type), reinterpret_cast<void**>(&d_data)))
	      throw Except("Failed to allocate GPU memory");
	  d_thrust_pointer =	thrust::device_ptr<element_type>(begin());
	}

	MatrixGPU(MatrixGPU &other);
	MatrixGPU(MatrixPtrGPU other, bool do_copy = true);

	MatrixGPU(Matrix &other);
	MatrixGPU(Matrix other);

	inline ~MatrixGPU() {
	  //std::cerr << "freeing " << size() << std::endl;
	  cublasFree(d_data);
	}

	inline element_type *ptr() {
		return d_data;
	}

	void set(std::vector<element_type> &v);
//	void set(float value);

	inline size_type &columns() {
		return d_columns;
	}

	inline size_type &rows() {
		return d_rows;
	}

	inline element_type *column_start(size_type index) {
		return &d_data[index * d_rows];
	}

//	inline element_type &operator[](size_type i) {
//		return d_data[i];
//	}
	thrust::device_ptr<element_type>::reference operator[](size_type i)
	{
		return *(d_thrust_pointer + i);
	}

	inline void allocate(std::vector<ptr_type *> &pointers) {
		size_type total_size(0);
		for (size_t i(0); i < pointers.size(); ++i) {
			total_size += pointers[i]->size();
		}
		
		//std::cerr << "allocating: " << size() << std::endl;
		if (CUBLAS_STATUS_SUCCESS != cublasAlloc(total_size, sizeof(element_type), reinterpret_cast<void**>(&d_data)))
		    throw Except("Couldnt allocate gpu matrix");
		d_rows = total_size;
		d_columns = 1;
		d_size = d_rows * d_columns;

		raw_ptr_type it(d_data);

		for (size_t i(0); i < pointers.size(); ++i) {
			pointers[i]->set_ptr(it);
			it += pointers[i]->size();
		}
	}

	inline element_type const &operator[](size_type i) const {
		return d_data[i];
	}

	void set(element_type value);
//	{
//		std::cout << "0-0" << begin() << " " << end() << std::endl;
//		thrust::device_ptr<element_type> a(begin());
//		thrust::device_ptr<element_type> b(end());
//
//		thrust::fill(a, b, 0.0);
//		std::cout << "0.0" << std::endl;
//	}

	void print();

	inline size_type size() const {return d_size;}

	iterator begin();

	iterator end();

	inline operator MatrixPtrGPU() {
		return MatrixPtrGPU(ptr(), d_rows, d_columns);
	}

	std::vector<element_type> to_vector();

	void clear();

	void fill_random(element_type std) {
	  ((ptr_type)(*this)).fill_random(std);
	}
		

	void fill_random_positive(element_type std) {
	  ((ptr_type)(*this)).fill_random_positive(std);
	}
};

class Matrix3DGPU {
public:
	typedef long int size_type;
	typedef MatrixPtrGPU ptr_type;

private:
	element_type *d_data;
	size_type d_rows, d_columns, d_slices;

public:
	typedef element_type *iterator;

	Matrix3DGPU(size_type rows, size_type columns, size_type slices) :
		d_data(0),
		d_rows(rows),
		d_columns(columns),
		d_slices(slices)
	{
	  //std::cerr << "allocating: " << size() << std::endl;
	  if (size())
	    if (CUBLAS_STATUS_SUCCESS != cublasAlloc(size(), sizeof(element_type), reinterpret_cast<void**>(&d_data)))
	      throw Except("Failed to allocate GPU memory");
	}

	Matrix3DGPU(Matrix3DGPU &other);
	Matrix3DGPU(Matrix3D &other);

	~Matrix3DGPU() {
	  //std::cerr << "freeing " << size() << std::endl;
	  cublasFree(reinterpret_cast<void**>(d_data));
	}

	element_type *ptr();

	void set(std::vector<element_type> &v);

	inline size_type &columns() const ;

	inline size_type &rows() const;

	inline size_type slices() const;

	iterator begin();

	iterator end();

	void print();

	inline size_type &columns() {
		return d_columns;
	}

	inline size_type &rows() {
		return d_rows;
	}

	inline size_type &slices() {
		return d_slices;
	}

	size_type size()  const;

	inline element_type &operator[](size_type i) {
		return d_data[i];
	}

	inline element_type const &operator[](size_type i) const {
		return d_data[i];
	}

	inline element_type *column_start(size_type index) {
		return &d_data[index * d_rows];
	}

	inline element_type *slice_start(size_type index) {
		return &d_data[index * d_rows * d_columns];
	}

	MatrixPtrGPU matrix_from_slice(size_type index);

	operator MatrixPtrGPU();

	std::vector<element_type> to_vector();

	void clear();
};

#endif /* MATRIX_H_ */

//#endif
