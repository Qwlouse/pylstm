#include "matrix.h"
#include <algorithm>

Matrix::Matrix(size_type _n_rows, size_type _n_columns, size_type _n_slices) :
  n_rows(_n_rows),
  n_columns(_n_columns),
  n_slices(_n_slices),
  data(NULL),
  n_size(n_rows * n_columns * n_slices)
{}


MatrixView2D::MatrixView2D(MatrixState _matrix_state, size_type _n_rows, size_type _n_columns, raw_ptr_type _data, size_type _stride) :
  matrix_state(_matrix_state),
  n_rows(_n_rows),
  n_columns(_n_columns),
  data(_data),
  stride(_stride)
{}

MatrixView3D::MatrixView3D(MatrixState _matrix_state, size_type _n_rows, size_type _n_columns, size_type _n_slices, raw_ptr_type _data, size_type _stride) :
  matrix_state(_matrix_state),
  n_rows(_n_rows),
  n_columns(_n_columns),
  n_slices(_n_slices),
  data(_data),
  stride(_stride)
{}
