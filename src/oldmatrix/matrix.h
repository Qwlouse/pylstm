/*
 * matrix.h
 *
 *  Created on: Jun 14, 2011
 *      Author: stollenga
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include <vector>
#include <algorithm>
#include <iostream>
#include <assert.h>
#include "defines.h"

class MatrixPtr {
public:
	typedef long int size_type;
	typedef element_type* ptr_type;

	ptr_type d_data_ptr;
	size_type d_rows;
	size_type d_columns;

public:
	typedef element_type *iterator;

	inline MatrixPtr(element_type *data_ptr, size_type rows, size_type columns) :
		d_data_ptr(data_ptr),
		d_rows(rows),
		d_columns(columns)
	{}

	inline MatrixPtr(size_type rows, size_type columns) :
		d_data_ptr(0),
		d_rows(rows),
		d_columns(columns)
	{}

	element_type *ptr() {
		return d_data_ptr;
	}

	element_type const *ptr() const {
		return d_data_ptr;
	}

	void set_ptr(element_type *ptr) {
		d_data_ptr = ptr;
	}

	size_type size() const {
		return d_rows * d_columns;
	}

	void set(std::vector<element_type> &v) {
		assert(size() == static_cast<size_type>(v.size()));
		std::copy(v.begin(), v.end(), d_data_ptr);
	}

	inline size_type &columns() {
		return d_columns;
	}

	inline size_type &rows() {
		return d_rows;
	}

	inline element_type *column_start(size_type index) {
		return &d_data_ptr[index * d_rows];
	}

	inline element_type &operator[](size_type i) {
		return d_data_ptr[i];
	}

	inline element_type const &operator[](size_type i) const {
		return d_data_ptr[i];
	}

	inline MatrixPtr operator+(int diff) {
		return MatrixPtr(ptr() + diff, d_rows, d_columns);
	}


	void print() const {
		for (size_type r(0); r < d_rows; ++r) {
			for (size_type c(0); c < d_columns; ++c)
				std::cout << d_data_ptr[c * d_rows + r] << " ";
			std::cout << std::endl;
		}
	}

	iterator begin() {
		return d_data_ptr;
	}

	iterator end() {
		return d_data_ptr + size();
	}

	size_type size() {
		return d_rows * d_columns;
	}

	std::vector<element_type> to_vector() {
		return std::vector<element_type>(d_data_ptr, d_data_ptr + size());
	}

	void clear() {
		std::fill(begin(), end(), 0.0);
	}

	void fill_random(element_type std) {
		size_type const MAX(100000);
		iterator it(begin()),
				end_it(end());

		for (; it != end_it; ++it)
			*it = ((element_type)(rand() % MAX)) / MAX * (2 * std) - std;
	}

	void fill_random(element_type std, element_type thresh) {
	  size_type const MAX(100000);
	  iterator it(begin()),
	    end_it(end());

	  for (; it != end_it; ++it) {
	    if(((element_type)(rand() % MAX)) / MAX > thresh)
	      *it = ((element_type)(rand() % MAX)) / MAX * (2 * std) - std;
	    else
	      *it = ((element_type)(rand() % MAX)) / MAX * (.01 * 2 * std) - .01 * std;
	  }

        }

	void fill_random_positive(element_type std) {
		size_type const MAX(100000);
		iterator it(begin()),
				end_it(end());

		for (; it != end_it; ++it)
			*it = ((element_type)(rand() % MAX)) / MAX * std;
	}

	void fill(element_type val) {
		iterator it(begin()),
				end_it(end());

		for (; it != end_it; ++it)
			*it = val;
	}

	void set_n(size_t n, element_type val) {
		iterator it(begin()), end_it(begin() + n);
		for (; it != end_it; ++it)
			*it = val;
	}

};

class Matrix {
public:
	typedef long int size_type;
	typedef MatrixPtr ptr_type;
	typedef element_type *raw_ptr_type;

private:
	std::vector<element_type> d_data;
	size_type d_rows;
	size_type d_columns;

public:
	typedef std::vector<element_type>::iterator iterator;

	Matrix(size_type rows = 0, size_type columns = 0) 
	  :
	d_data(columns * rows),
	  d_rows(rows),
	  d_columns(columns)
	{}

	Matrix(std::vector<element_type> v) 
	  :
	d_data(v),
	  d_rows(1),
	  d_columns(d_data.size())
	{}

	Matrix(MatrixPtr &from, bool do_copy = true);

	Matrix(std::vector<element_type>::iterator v_start, std::vector<element_type>::iterator v_end) 
	  :
	d_data(v_start, v_end),
	  d_rows(1),
	  d_columns(d_data.size())
	    {}
	
	void resize(size_type rows, size_type columns) {
	  d_rows = rows;
	  d_columns = columns;
	  d_data.resize(d_rows * d_columns);
	}

	element_type *ptr() {
		return &d_data[0];
	}

	void set(std::vector<element_type> &v) {
		assert(d_data.size() == v.size());
		std::copy(v.begin(), v.end(), d_data.begin());
	}

	inline size_type columns() const {
		return d_columns;
	}

	inline size_type rows() const {
		return d_rows;
	}

	inline element_type *column_start(size_type index) {
		return &d_data[index * d_rows];
	}

	inline element_type &operator[](size_type i) {
		return d_data[i];
	}

	inline element_type const &operator[](size_type i) const {
		return d_data[i];
	}

	inline element_type* it(size_type i) {
		return &d_data[0] + i;
	}

	inline element_type const* it(size_type i) const {
		return &d_data[0] + i;
	}

	inline void allocate(std::vector<ptr_type *> &pointers) {
		size_type total_size(0);
		for (size_t i(0); i < pointers.size(); ++i) {
			total_size += pointers[i]->size();
		}
		d_data.resize(total_size);
		d_rows = total_size;
		d_columns = 1;

		raw_ptr_type it(&d_data[0]);

		for (size_t i(0); i < pointers.size(); ++i) {
			pointers[i]->set_ptr(it);
			it += pointers[i]->size();
		}
	}

	void print() const {
		std::cout << "matrix:" << std::endl;
		for (size_type r(0); r < d_rows; ++r) {
			for (size_type c(0); c < d_columns; ++c)
				std::cout << d_data[c * d_rows + r] << " ";
			std::cout << std::endl;
		}
	}

	size_type size() const {
		return d_rows * d_columns;
	}

	iterator begin() {
		return d_data.begin();
	}

	iterator end() {
		return d_data.end();
	}

	operator ptr_type() {
		return ptr_type(ptr(), d_rows, d_columns);
	}

	std::vector<element_type> to_vector() {
		return d_data;
	}

	void clear() {
		std::fill(begin(), end(), 0.0);
	}

	void fill_random(element_type std) {
	  ((ptr_type)(*this)).fill_random(std);
	}

	void fill_random_positive(element_type std) {
	  ((ptr_type)(*this)).fill_random_positive(std);
	}

	void set(element_type value) {
		iterator it(begin()),
				end_it(end());

		for (; it != end_it; ++it)
			*it = value;
	}

};

class Matrix3D {
public:
	typedef long int size_type;
	typedef MatrixPtr ptr_type;

private:
	std::vector<element_type> d_data;
	size_type d_rows, d_columns, d_slices;

public:
	typedef std::vector<element_type>::iterator iterator;

	Matrix3D(size_type rows = 0, size_type columns = 0, size_type slices = 0) :
		d_data(columns * rows * slices),
		d_rows(rows),
		d_columns(columns),
		d_slices(slices)
	{
	}

	element_type *ptr() {
		return &d_data[0];
	}

	void set(std::vector<element_type> &v) {
		std::copy(v.begin(), v.end(), d_data.begin());
	}

	inline size_type &columns() {
		return d_columns;
	}

	inline size_type &rows() {
		return d_rows;
	}

	inline size_type &slices() {
		return d_slices;
	}

	iterator begin() {
		return d_data.begin();
	}

	iterator end() {
		return d_data.end();
	}

	void print() const {
		for (size_type s(0); s < d_slices; ++s) {
			std::cout << ">";
			for (size_type r(0); r < d_rows; ++r) {
				for (size_type c(0); c < d_columns; ++c)
					std::cout << d_data[s * d_rows * d_columns + c * d_rows + r] << " ";
				std::cout << std::endl;
			}
		}
	}

	inline size_type size() const {
		return d_rows * d_columns * d_slices;
	}

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

	MatrixPtr matrix_from_slice(size_type index) {
		return MatrixPtr(slice_start(index), d_rows, d_columns);
	}

	operator MatrixPtr() {
		return MatrixPtr(ptr(), d_rows, d_columns * d_slices);
	}

	std::vector<element_type> to_vector() {
		return d_data;
	}

	void clear() {
		std::fill(begin(), end(), 0.0);
	}
};

#endif /* MATRIX_H_ */
