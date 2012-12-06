#include "matrix.h"
#include <algorithm>

Matrix::Matrix(size_type _n_rows, size_type _n_columns, size_type _n_slices) :
  n_rows(_n_rows),
  n_columns(_n_columns),
  n_slices(_n_slices),
  data(NULL),
  size(n_rows * n_columns * n_slices)
{}


MatrixView2D::MatrixView2D(MatrixState _state, size_type _n_rows, size_type _n_columns, raw_ptr_type _data, size_type _stride) :
  state(_state),
  n_rows(_n_rows),
  n_columns(_n_columns),
  data(_data),
  size(n_rows * n_columns),
  stride(_stride)
{}

MatrixView3D::MatrixView3D(MatrixState _state, size_type _n_rows, size_type _n_columns, size_type _n_slices, raw_ptr_type _data, size_type _stride) :
  state(_state),
  n_rows(_n_rows),
  n_columns(_n_columns),
  n_slices(_n_slices),
  data(_data),
  size(n_rows * n_columns * n_slices),
  stride(_stride)
{}
