#include "matrix_cpu.h"
#include "Core.h"

#include <algorithm>
#include <iostream>

using namespace std;

MatrixCPU::MatrixCPU(size_type _n_rows, size_type _n_columns, size_type _n_slices) :
  Matrix(_n_rows, _n_columns, _n_slices)
{
    allocate();
	std::fill(data, data + size, 0);
    standard_view_2d = MatrixView2DCPU(NORMAL, n_rows, n_columns, data, 0);
    standard_view_3d = MatrixView3DCPU(NORMAL, n_rows, n_columns, n_slices, data, 0);
}

MatrixCPU::MatrixCPU(initializer_list<initializer_list<double>> values) :
  Matrix(values.size(), values.begin()->size(), 1) {
  allocate();
  std::fill(data, data + size, 0);
  standard_view_2d = MatrixView2DCPU(NORMAL, n_rows, n_columns, data, 0);
  standard_view_3d = MatrixView3DCPU(NORMAL, n_rows, n_columns, n_slices, data, 0);
  
  
  for (initializer_list<initializer_list<double>>::const_iterator it(values.begin()); it != values.end(); ++it) {
    cout << it->size() << " " << n_rows << endl;
    ASSERT(it->size() == n_columns);
    for (initializer_list<double>::const_iterator it2(it->begin()); it2 != it->end(); ++it2) {
      size_t row(it - values.begin()), col(it2 - it->begin());
      cout << row << " " << col << endl;
      *(data + n_rows * col + row) = *it2;
    }
  }
}

MatrixCPU::MatrixCPU(d_type* _data, size_type _n_rows, size_type _n_columns, size_type _n_slices) :
    Matrix(_data, _n_rows, _n_columns, _n_slices) {
  standard_view_2d = MatrixView2DCPU(NORMAL, n_rows, n_columns, data, 0);
  standard_view_3d = MatrixView3DCPU(NORMAL, n_rows, n_columns, n_slices, data, 0);
}

void MatrixCPU::allocate() {
  cout << "allocating!" << endl;
  data = new d_type[size];	
}

MatrixCPU::~MatrixCPU() {
	delete[] data;
}


//VIEWS
MatrixView3DCPU::MatrixView3DCPU(MatrixState _matrix_state, size_type _n_rows, size_type _n_columns, size_type _n_slices, raw_ptr_type _data, size_type _stride) :
  MatrixView3D(_matrix_state, _n_rows, _n_columns, _n_slices, _data, _stride),
  matrix_view_2d(NORMAL, _n_rows, _n_columns * _n_slices, data, 0)
{}

MatrixView3DCPU::MatrixView3DCPU() :
  MatrixView3D(NORMAL, 0, 0, 0, 0, 0),
  matrix_view_2d(NORMAL, 0, 0, 0, 0)
{}

MatrixView2DCPU &MatrixView3DCPU::flatten() {
    return matrix_view_2d;
}

MatrixView2DCPU MatrixView3DCPU::slice(size_type t) {
  return MatrixView2DCPU(state, n_rows, n_columns, data + n_rows * n_columns * t, 0);
}

///Matrix View 2d cpu
MatrixView2DCPU::MatrixView2DCPU(MatrixState _matrix_state, size_type _n_rows, size_type _n_columns, raw_ptr_type _data, size_type _stride) :
  MatrixView2D(_matrix_state, _n_rows, _n_columns, _data, _stride)
{}

MatrixView2DCPU::MatrixView2DCPU() :
  MatrixView2D(NORMAL, 0, 0, 0, 0)
{}


std::ostream &operator<<(std::ostream &out, MatrixCPU &in) {
  for (size_t i(0); i < in.size; ++i) {
    out << in.data[i] << " ";
  }
  return out;
}
