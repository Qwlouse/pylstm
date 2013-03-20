#include "matrix.h"
#include <algorithm>
#include "Core.h"
#include <iostream>
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
	stride(0),
	data(NULL),
	state(NORMAL),
	n_rows(0),
	n_columns(0),
	n_slices(0),
	size(0)
{ }

Matrix::Matrix(const data_ptr data, const size_t offset, const int stride, const MatrixState state, const size_t n_rows, const size_t n_columns, const size_t n_slices) :
	offset(offset),
	stride(stride),
	data(data),
	state(state),
	n_rows(n_rows),
	n_columns(n_columns),
	n_slices(n_slices),
	size(n_rows * n_columns * n_slices)
{ }

Matrix::Matrix(size_t n_rows, size_t n_columns, size_t n_slices, MatrixState state) :
	offset(0),
	stride(0),
	data(new d_type[n_rows * n_columns * n_slices]),
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
	stride(0),
	data(new d_type[values.size()]),
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
		stride(0),
		data(new d_type[values.size() * values.begin()->size()]),
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
		stride(0),
		data(new d_type[values.size() * values.begin()->size() * values.begin()->begin()->size()]),
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
		stride(0),
		data(data_ptr, NullDeleter<d_type>()),
		state(NORMAL),
		n_rows(n_rows),
		n_columns(n_columns),
		n_slices(n_slices),
		size(n_rows * n_columns * n_slices)
{
}


d_type& Matrix::operator[](size_t index) {
	ASSERT(index < size);
	return data[offset + index];
}

size_t Matrix::get_offset(size_t row, size_t col, size_t slice)
{
	ASSERT(row < n_rows);
	ASSERT(col < n_columns);
	ASSERT(slice < n_slices);
	if (state == NORMAL) {
		return offset + slice*(n_rows + stride)*n_columns + col*(n_rows + stride) + row;
	} else {
		return offset + slice*(n_rows + stride)*n_columns + row*n_columns + col;
	}
}

d_type& Matrix::get(size_t row, size_t col, size_t slice)
{
	return data[get_offset(row, col, slice)];
}



Matrix Matrix::slice(size_t slice_index)
{
	return Matrix(data, get_offset(0, 0, slice_index), stride, state, n_rows, n_columns, 1);
}

Matrix Matrix::slice(size_t start, size_t stop)
{
	return Matrix(data, get_offset(0, 0, start), stride, state, n_rows, n_columns, stop - start + 1);
}

Matrix Matrix::row_slice(size_t row_index)
{
    ASSERT(stride == 0);
    return Matrix(data, get_offset(row_index, 0, 0), n_rows - 1, state, 1, n_columns, n_slices);
}


Matrix Matrix::subslice(size_t start, size_t rows, size_t columns, size_t slices)
{
	return Matrix(data, offset + start, stride, state, rows, columns, slices);
}

Matrix Matrix::T() {
	return Matrix(data, offset, stride, transpose(state), n_columns, n_rows, n_slices);
}

Matrix Matrix::flatten_time() {
	return Matrix(data, offset, stride, state, n_rows, n_columns * n_slices, 1);
}

void Matrix::set_all_elements_to(d_type value) {
	for (int i = 0; i < size; ++i)
		operator[](i) = value;
}

void Matrix::print_me() {
  cout << "Matrix 3D: " << n_rows << " x " << n_columns << " x " << n_slices << " # " << offset << '\n';

  d_type* data_ptr = get_data();
  cout << "=====================================\n";
  for (int s = 0; s < n_slices; ++s) {
    for (int r = 0; r < n_rows; ++r) {
      for (int c = 0; c < n_columns; ++c) {
        cout << *data_ptr << " ";
        data_ptr += n_rows;
      }
      data_ptr -= n_columns*n_rows - 1;
      cout << '\n';
    }
    data_ptr += (n_columns-1)*n_rows;
    cout << "=====================================\n";
  }
}


