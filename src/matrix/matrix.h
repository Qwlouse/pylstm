#pragma once
#include <boost/shared_array.hpp>
#include <cstddef>
#include <vector>
#include <initializer_list>

enum MatrixState {
	NORMAL,
	TRANSPOSED
};

inline MatrixState transpose(MatrixState s) {
	return s == NORMAL ? TRANSPOSED : NORMAL;
};

typedef double d_type;
using std::size_t;


typedef boost::shared_array<d_type> data_ptr;

class Matrix {
private:
	size_t offset;
	int stride;
	data_ptr data;
public:
	MatrixState state;
	size_t n_rows;
	size_t n_columns;
	size_t n_slices;
	size_t size;

	/// default constructor for Matrix of size 0
	Matrix();
	/// initializer list constructors
	Matrix(std::initializer_list<d_type> values);
	Matrix(std::initializer_list<std::initializer_list<d_type>> values);
	Matrix(std::initializer_list<std::initializer_list<std::initializer_list<d_type>>> values);
	/// full constructor that takes a shared data pointer and all other parameters
	Matrix(const data_ptr data, const size_t offset, const int stride, const MatrixState state, const size_t n_rows, const size_t n_columns, const size_t n_slices);
	/// allocating contstructor that takes sizes and does allocate the array
	Matrix(size_t n_rows, size_t n_columns, size_t n_slices, MatrixState state=NORMAL);
	/// constructor taking a raw data_ptr and sizes, so this Matrix does not delete the data ever
	Matrix(d_type* data_ptr, size_t n_rows, size_t n_columns, size_t n_slices);

	virtual ~Matrix() { };

	d_type &operator[](size_t index);
	d_type& get(size_t row, size_t col, size_t slice);
	size_t get_offset(size_t row, size_t col, size_t slice);

	inline d_type* get_data() {return &data[offset];}

	Matrix subslice(size_t start, size_t n_rows, size_t n_columns, size_t n_slices); // todo: rename

	Matrix slice(size_t slice_index);
	Matrix slice(size_t start, size_t stop);
	Matrix row_slice(size_t row_index);
	Matrix row_slice(size_t start_row, size_t stop_row);

	Matrix T();
	Matrix flatten_time();
	void set_all_elements_to(d_type value);

	void print_me();
};


