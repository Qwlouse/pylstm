#include "matrix.h"
#include <algorithm>
#include "Core.h"

Matrix::Matrix(const data_ptr data, const size_t offset, const MatrixState state, const size_t n_rows, const size_t n_columns, const size_t n_slices) :
	state(state),
	offset(offset),
	data(data),
	n_rows(n_rows),
	n_columns(n_columns),
	n_slices(n_slices),
	size(n_rows * n_columns * n_slices)
{ }

Matrix::Matrix(size_t n_rows, size_t n_columns, size_t n_slices, MatrixState state) :
	state(state),
	offset(0),
	data(new d_type[n_rows * n_columns * n_slices]),
	n_rows(n_rows),
	n_columns(n_columns),
	n_slices(n_slices),
	size(n_rows * n_columns * n_slices)
{
	std::fill_n(&data[0], size, 0.0);
}

Matrix::Matrix(std::initializer_list<d_type> values):
	state(NORMAL),
	offset(0),
	data(new d_type[values.size()]),
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
		state(NORMAL),
		offset(0),
		data(new d_type[values.size() * values.begin()->size()]),
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
		state(NORMAL),
		offset(0),
		data(new d_type[values.size() * values.begin()->size() * values.begin()->begin()->size()]),
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

d_type& Matrix::get(size_t row, size_t col, size_t slice)
{
	ASSERT(row < n_rows);
	ASSERT(col < n_columns);
	ASSERT(slice < n_slices);
	if (state == NORMAL) {
		return data[slice*n_rows*n_columns + col*n_rows + row];
	} else {
		return data[slice*n_rows*n_columns + row*n_columns + col];
	}
}

Matrix Matrix::T() {
	return Matrix(data, offset, transpose(state), n_columns, n_rows, n_slices);
}
