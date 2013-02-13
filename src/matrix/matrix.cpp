#include "matrix.h"
#include <algorithm>

Matrix::Matrix(const data_ptr data, const size_t offset, const MatrixState state, const size_t n_rows, const size_t n_columns, const size_t n_slices) :
	state(state),
	offset(offset),
	data(data),
	n_rows(n_rows),
	n_columns(n_columns),
	n_slices(n_slices),
	size(n_rows * n_columns * n_slices)
{ }

Matrix::Matrix(MatrixState state, size_t n_rows, size_t n_columns, size_t n_slices) :
	state(state),
	offset(0),
	data(new d_type[10]),
	n_rows(n_rows),
	n_columns(n_columns),
	n_slices(n_slices),
	size(n_rows * n_columns * n_slices)
{ }
