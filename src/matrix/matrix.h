#pragma once
#include <boost/shared_array.hpp>
#include <cstddef>
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
	const MatrixState state;
	const size_t offset;
	const data_ptr data;
public:
	const size_t n_rows;
	const size_t n_columns;
	const size_t n_slices;
	const size_t size;
	Matrix(std::initializer_list<d_type> values);
	Matrix(std::initializer_list<std::initializer_list<d_type>> values);
	Matrix(std::initializer_list<std::initializer_list<std::initializer_list<d_type>>> values);
	Matrix(const data_ptr data, const size_t offset, const MatrixState state, const size_t n_rows, const size_t n_columns, const size_t n_slices);
	Matrix(size_t n_rows, size_t n_columns, size_t n_slices, MatrixState state=NORMAL);
	virtual ~Matrix() { };

	inline d_type &operator[](size_t index) {return data[index];}
	d_type& get(size_t row, size_t col, size_t slice);
	inline d_type* get_data() {return &data[0];}
	Matrix T();
};


