/*
* matrix.h
*
*  Created on: Jun 14, 2011
*      Author: stollenga
*/
#include <assert.h>

#include "matrix_cuda.h"
#include "matrix.h"
#include "matrix_operations_cuda.h"


//	void MatrixPtrGPU::set(std::vector<element_type> &v) {
//		assert(size() == v.size());
//		std::copy(v.begin(), v.end(), d_data_ptr);
//	}

//	void MatrixPtrGPU::print() {
//		for (size_t r(0); r < d_rows; ++r) {
//			for (size_t c(0); c < d_columns; ++c)
//				std::cout << d_data_ptr[c * d_rows + r] << " ";
//			std::cout << std::endl;
//		}
//	}

MatrixPtrGPU::iterator MatrixPtrGPU::begin() {
	return d_data_ptr;
}

MatrixPtrGPU::iterator MatrixPtrGPU::end() {
	return d_data_ptr + size();
}

std::vector<element_type> MatrixPtrGPU::to_vector() {
	Matrix temp(d_rows, d_columns);
	copy(*this, temp);
	return temp.to_vector();
}

void MatrixPtrGPU::fill_random(element_type std) {
	Matrix temp(d_rows, d_columns);
	temp.fill_random(std);
	copy(temp, *this);
}

void MatrixPtrGPU::fill_random_positive(element_type std) {
	Matrix temp(d_rows, d_columns);
	temp.fill_random_positive(std);
	copy(temp, *this);
}


void MatrixPtrGPU::set(element_type value) {
	Matrix value_mat(1, 1);
	value_mat[0] = value;
	MatrixGPU gpu_value_mat(1, 1);
	copy(value_mat, gpu_value_mat);

	cublasDcopy(size(), gpu_value_mat.ptr(), 0, d_data_ptr, 1);
}


void MatrixPtrGPU::set_n(size_t n, element_type value) {
	Matrix value_mat(1, 1);
	value_mat[0] = value;
	MatrixGPU gpu_value_mat(1, 1);
	copy(value_mat, gpu_value_mat);

	cublasDcopy(n, gpu_value_mat.ptr(), 0, d_data_ptr, 1);
}

void MatrixPtrGPU::clear() {
	set(0.0);
}


MatrixGPU::MatrixGPU(MatrixGPU &other) :
	d_data(0),
	d_rows(other.rows()),
	d_columns(other.columns()),
	d_size(d_rows * d_columns)
{
	cublasAlloc(size(), sizeof(element_type), reinterpret_cast<void**>(&d_data));
	copy(other, *this);
}

MatrixGPU::MatrixGPU(MatrixPtrGPU other, bool do_copy) :
	d_data(0),
	d_rows(other.rows()),
	d_columns(other.columns()),
	d_size(d_rows * d_columns)
{
//	std::cout << ----------------"mgpu is called " << size() << std::endl;
	cublasAlloc(size(), sizeof(element_type), reinterpret_cast<void**>(&d_data));
	if (do_copy)
		copy(other, *this);
	else
		set(0.0);
}


MatrixGPU::MatrixGPU(Matrix &other) :
	d_data(0),
	d_rows(other.rows()),
	d_columns(other.columns()),
	d_size(d_rows * d_columns)
{
	cublasAlloc(size(), sizeof(element_type), reinterpret_cast<void**>(&d_data));
	copy(other, *this);
}

MatrixGPU::MatrixGPU(Matrix other) :
	d_data(0),
	d_rows(other.rows()),
	d_columns(other.columns()),
	d_size(d_rows * d_columns)
{
	cublasAlloc(size(), sizeof(element_type), reinterpret_cast<void**>(&d_data));
	copy(other, *this);
}

void MatrixGPU::set(std::vector<element_type> &v) {
	assert(static_cast<size_type>(v.size()) == size());
	cublasDcopy(v.size(), &v[0], 1, d_data, 1);
}

void MatrixGPU::set(element_type value) {
	Matrix value_mat(1, 1);
	value_mat[0] = value;
	MatrixGPU gpu_value_mat(1, 1);
	copy(value_mat, gpu_value_mat);

	cublasDcopy(size(), gpu_value_mat.ptr(), 0, d_data, 1);
}

//	void MatrixGPU::print() {
//		for (size_t r(0); r < d_rows; ++r) {
//			for (size_t c(0); c < d_columns; ++c)
//				std::cout << d_data[c * d_rows + r] << " ";
//			std::cout << std::endl;
//		}
//	}


MatrixGPU::iterator MatrixGPU::begin() {
	return d_data;
}

MatrixGPU::iterator MatrixGPU::end() {
	return d_data + size();
}


//	MatrixGPU:operator std::vector<element_type>() {
//		return d_data;
//	}

void MatrixGPU::clear() {
	 cudaMemset(d_data, 0, size() * sizeof(element_type));
}

/*
void MatrixGPU::fill_random(element_type std) {
	Matrix temp(d_rows, d_columns);
	temp.fill_random(std);
	copy(temp, *this);
	}*/

std::vector<element_type> MatrixGPU::to_vector() {
	Matrix temp(d_rows, d_columns);
	copy(*this, temp);
	return temp.to_vector();
}

void MatrixGPU::print() {
	Matrix temp(d_rows, d_columns);
	copy(*this, temp);
	temp.print();
}




//	void MatrixGPU:fill_random(element_type std) {
//		size_t const MAX(100000);
//		iterator it(begin()),
//				end_it(end());
//
//		for (; it != end_it; ++it)
//			*it = ((element_type)(rand() % MAX)) / MAX * (2 * std) - std;
//	}



Matrix3DGPU::Matrix3DGPU(Matrix3DGPU &other) :
	d_data(0),
	d_rows(other.d_rows),
	d_columns(other.d_columns),
	d_slices(other.d_slices)
{
	cublasAlloc(size(), sizeof(element_type), reinterpret_cast<void**>(&d_data));
	copy(other, *this);
}

Matrix3DGPU::Matrix3DGPU(Matrix3D &other) :
	d_data(0),
	d_rows(other.rows()),
	d_columns(other.columns()),
	d_slices(other.slices())
{
	cublasAlloc(size(), sizeof(element_type), reinterpret_cast<void**>(&d_data));
	copy(other, *this);
}

element_type *Matrix3DGPU::ptr() {
	return &d_data[0];
}

void Matrix3DGPU::set(std::vector<element_type> &v) {
	assert(static_cast<size_type>(v.size()) == size());
	cublasDcopy(v.size(), &v[0], 1, d_data, 1);
}


Matrix3DGPU::iterator Matrix3DGPU::begin() {
	return d_data;
}

Matrix3DGPU::iterator Matrix3DGPU::end() {
	return d_data + size();
}

//	void Matrix3DGPU::print() {
//		for (size_t s(0); s < d_slices; ++s) {
//			std::cout << ">";
//			for (size_t r(0); r < d_rows; ++r) {
//				for (size_t c(0); c < d_columns; ++c)
//					std::cout << d_data[s * d_rows * d_columns + c * d_rows + r] << " ";
//				std::cout << std::endl;
//			}
//		}
//	}

Matrix3DGPU::size_type Matrix3DGPU::size() const {
	return d_rows * d_columns * d_slices;
}

MatrixPtrGPU Matrix3DGPU::matrix_from_slice(size_type index) {
	return MatrixPtrGPU(slice_start(index), d_rows, d_columns);
}

Matrix3DGPU::operator MatrixPtrGPU() {
	return MatrixPtrGPU(ptr(), d_rows, d_columns * d_slices);
}

std::vector<element_type> Matrix3DGPU::to_vector() {
	Matrix temp(d_rows, d_columns * d_slices);
	copy(*this, temp);
	return temp.to_vector();
}

//
//	Matrix3DGPU::operator std::vector<element_type>() {
//		return d_data;
//	}

void Matrix3DGPU::clear() {
	 cudaMemset(d_data, 0, size() * sizeof(element_type));
}

void Matrix3DGPU::print() {
	Matrix3D temp(d_rows, d_columns, d_slices);
	copy(*this, temp);
	temp.print();
}

