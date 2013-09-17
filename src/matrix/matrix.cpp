#include "matrix.h"

#include <algorithm>
#include <iostream>

#include "Core.h"

using std::cout;

template<typename T>
struct NullDeleter
{
   void operator()(T* )
   {
      // do nothing
   }
};

Matrix::Matrix() :
	offset(0),
	data(),
	stride(0),
	state(NORMAL),
	n_rows(0),
	n_columns(0),
	n_slices(0),
	size(0)
{ }

Matrix::Matrix(const data_ptr data, const size_t offset, const int stride, const MatrixState state, const size_t n_rows, const size_t n_columns, const size_t n_slices) :
	offset(offset),
	data(data),
	stride(stride),
	state(state),
	n_rows(n_rows),
	n_columns(n_columns),
	n_slices(n_slices),
	size(n_rows * n_columns * n_slices)
{ }

Matrix::Matrix(size_t n_rows, size_t n_columns, size_t n_slices, MatrixState state) :
	offset(0),
	data(new d_type[n_rows * n_columns * n_slices]),
	stride(0),
	state(state),
	n_rows(n_rows),
	n_columns(n_columns),
	n_slices(n_slices),
	size(n_rows * n_columns * n_slices)
{
	std::fill_n(&data[0], size, 0.0);
}

Matrix::Matrix(std::initializer_list<d_type> values):
	offset(0),
	data(new d_type[values.size()]),
	stride(0),
	state(NORMAL),
	n_rows(1),
	n_columns(values.size()),
	n_slices(1),
	size(values.size())
{
	for (std::initializer_list<double>::const_iterator it(values.begin()); it != values.end(); ++it) {
		size_t row(it - values.begin());
		data[row] = *it;
	}
}


Matrix::Matrix(std::initializer_list<std::initializer_list<double>> values) :
		offset(0),
		data(new d_type[values.size() * values.begin()->size()]),
		stride(0),
		state(NORMAL),
		n_rows(values.size()),
		n_columns(values.begin()->size()),
		n_slices(1),
		size(values.size() * values.begin()->size())
{
  for (std::initializer_list<std::initializer_list<double>>::const_iterator it(values.begin()); it != values.end(); ++it) {
    ASSERT(it->size() == n_columns);
    for (std::initializer_list<double>::const_iterator it2(it->begin()); it2 != it->end(); ++it2) {
      size_t row(it - values.begin()), col(it2 - it->begin());
      data[n_rows * col + row] = *it2;
    }
  }
}

Matrix::Matrix(std::initializer_list<std::initializer_list<std::initializer_list<double>>> values) :
		offset(0),
		data(new d_type[values.size() * values.begin()->size() * values.begin()->begin()->size()]),
		stride(0),
		state(NORMAL),
		n_rows(values.begin()->size()),
		n_columns(values.begin()->begin()->size()),
		n_slices(values.size()),
		size(values.size() * values.begin()->size() * values.begin()->begin()->size())
{
	for (std::initializer_list<std::initializer_list<std::initializer_list<double>>>::const_iterator it(values.begin()); it != values.end(); ++it) {
		size_t slice(it - values.begin());
		ASSERT(it->size() == n_rows);
		for (std::initializer_list<std::initializer_list<double>>::const_iterator it2(it->begin()); it2 != it->end(); ++it2) {
			ASSERT(it2->size() == n_columns);
			size_t row(it2 - it->begin());
			for (std::initializer_list<double>::const_iterator it3(it2->begin()); it3 != it2->end(); ++it3) {
				size_t col(it3 - it2->begin());
				get(row, col, slice) = *it3;
			}
		}
	}
}

Matrix::Matrix(d_type* data_ptr, size_t n_rows, size_t n_columns, size_t n_slices) :
		offset(0),
		data(data_ptr, NullDeleter<d_type>()),
		stride(0),
		state(NORMAL),
		n_rows(n_rows),
		n_columns(n_columns),
		n_slices(n_slices),
		size(n_rows * n_columns * n_slices)
{
}


d_type& Matrix::operator[](size_t index) const {
	ASSERT(index < size);
	if (state == NORMAL) {
	    if (stride == 0) {
	        return data[offset + index];
	    }
        size_t i = (index / n_rows) * (n_rows + stride) + (index % n_rows);
        return data[offset + i];
	}
    size_t slice = index / (n_rows * n_columns);
    size_t col = (index % (n_rows * n_columns)) / n_rows;
    size_t row = (index % (n_rows * n_columns)) % n_rows;

    return get(row, col, slice);
}

size_t Matrix::get_offset(size_t row, size_t col, size_t slice) const
{
	ASSERT(row < n_rows);
	ASSERT(col < n_columns);
	ASSERT(slice < n_slices);
	if (state == NORMAL) {
		return offset + slice*(n_rows + stride)*n_columns + col*(n_rows + stride) + row;
	} else {
		return offset + slice*(n_columns + stride)*n_rows + row*(n_columns + stride) + col;
	}
}

d_type& Matrix::get(size_t row, size_t col, size_t slice) const
{
	return data[get_offset(row, col, slice)];
}



Matrix Matrix::slice(size_t slice_index)
{
	return Matrix(data, get_offset(0, 0, slice_index), stride, state, n_rows, n_columns, 1);
}

Matrix Matrix::slice(size_t start, size_t stop)
{
	return Matrix(data, get_offset(0, 0, start), stride, state, n_rows, n_columns, stop - start);
}

Matrix Matrix::row_slice(size_t row_index)
{
    ASSERT(stride == 0);
    ASSERT(state == NORMAL);
    return Matrix(data, get_offset(row_index, 0, 0), static_cast<int>(n_rows - 1), state, 1, n_columns, n_slices);
}

Matrix Matrix::row_slice(size_t start_row, size_t stop_row)
{
    ASSERT(stride == 0);
    ASSERT(state == NORMAL);
    size_t rows = (stop_row - start_row);
    return Matrix(data, get_offset(start_row, 0, 0), static_cast<int>(n_rows - rows), state, rows, n_columns, n_slices);
}


Matrix Matrix::sub_matrix(size_t start, size_t rows, size_t columns, size_t slices)
{
    ASSERT(stride == 0);
	return Matrix(data, offset + start, 0, state, rows, columns, slices);
}

Matrix Matrix::T() {
	return Matrix(data, offset, stride, transpose(state), n_columns, n_rows, n_slices);
}

Matrix Matrix::flatten_time() {
    ASSERT(stride == 0);
	return Matrix(data, offset, 0, state, n_rows, n_columns * n_slices, 1);
}

void Matrix::set_all_elements_to(d_type value) {
	for (d_type& v : *this)
		v = value;
}

void Matrix::print_me() {
  cout << "Matrix 3D: " << n_rows << " x " << n_columns << " x " << n_slices << " # " << offset << '\n';

  cout << "=====================================\n";
  for (int s = 0; s < n_slices; ++s) {
    for (int r = 0; r < n_rows; ++r) {
      for (int c = 0; c < n_columns; ++c) {
        cout << get(r, c, s) << " ";
      }
      cout << '\n';
    }
    cout << "=====================================\n";
  }
}


// iterator implementation

Matrix::iterator::iterator(Matrix* m_) :
    ptr(m_->get_data()),
    i(0),
    m(m_)
{ }

Matrix::iterator::iterator(Matrix* m_, size_t i) :
    ptr(i < m_->size ? &(*m_)[i] : NULL),
    i(i),
    m(m_)
{ }

Matrix::iterator::~iterator()
{ }


Matrix::iterator& Matrix::iterator::operator++() {
    ++i;
    if (i >= m->size) {
        ptr = NULL;
    } else {
        ptr = &((*m)[i]);
    }
    return *this;
}

d_type& Matrix::iterator::operator*() {
    return *ptr;
}

bool Matrix::iterator::operator==(const Matrix::iterator& o) const {
    return o.ptr == ptr;
}

bool Matrix::iterator::operator!=(const Matrix::iterator& o) const {
    return o.ptr != ptr;
}


Matrix::iterator Matrix::begin() {
    return Matrix::iterator(this);
}

Matrix::iterator Matrix::end() {
    return Matrix::iterator(this, size);
}



