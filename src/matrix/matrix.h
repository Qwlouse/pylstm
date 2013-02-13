#pragma once
#include <boost/shared_array.hpp>
#include <cstddef>

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

	Matrix(const data_ptr data, const size_t offset, const MatrixState state, const size_t n_rows, const size_t n_columns, const size_t n_slices);
	Matrix(MatrixState state, size_t n_rows, size_t n_columns, size_t n_slices);
	virtual ~Matrix() { };

	inline d_type &operator[](size_t index) {return data[index];}
	inline d_type* get_data() {return &data[0];}
};


