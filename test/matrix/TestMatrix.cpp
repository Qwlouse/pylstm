#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "Config.h"

#include "matrix/matrix.h"
#include "matrix/matrix_operation.h"

class MatrixTest : public ::testing::Test {
protected:
  MatrixTest() :
    M({{{0, 1, 2, 3}, {10, 11, 12, 13}, {20, 21, 22, 23}},
        {{100, 101, 102, 103}, {110, 111, 112, 113}, {120, 121, 122, 123}}})
{  }

    Matrix M;
};


TEST_F(MatrixTest, check_1D_initializer_list)
{
	Matrix m = {0, 2, 4, 6};
	ASSERT_EQ(m.n_rows, 1);
	ASSERT_EQ(m.n_columns, 4);
	ASSERT_EQ(m.n_slices, 1);
	for (int i = 0; i < 4; ++i) {
		ASSERT_EQ(m.get(0, i, 0), 2*i);
	}
}

TEST_F(MatrixTest, check_2D_initializer_list)
{
	Matrix m = {{0, 2, 4, 6},
			{10, 12, 14, 16}};
	ASSERT_EQ(m.n_rows, 2);
	ASSERT_EQ(m.n_columns, 4);
	ASSERT_EQ(m.n_slices, 1);
	for (int r = 0; r < 2; ++r) {
		for (int c = 0; c < 4; ++c) {
			ASSERT_EQ(m.get(r, c, 0), 2*c + 10*r);
		}
	}
}

TEST_F(MatrixTest, check_3D_initializer_list)
{
	Matrix m = {{{0, 1, 2, 3}, {10, 11, 12, 13}, {20, 21, 22, 23}},
			{{100, 101, 102, 103}, {110, 111, 112, 113}, {120, 121, 122, 123}}};
	ASSERT_EQ(m.n_rows, 3);
	ASSERT_EQ(m.n_columns, 4);
	ASSERT_EQ(m.n_slices, 2);
	for (int s = 0; s < 2; ++s) {
		for (int r = 0; r < 3; ++r) {
			for (int c = 0; c < 4; ++c) {
				ASSERT_EQ(m.get(r, c, s), c + 10*r + 100*s);
			}
		}
	}
}


TEST_F(MatrixTest, check_indexing_operator_standard)
{
    std::vector<int> expected = {0, 10, 20, 1, 11, 21, 2, 12, 22, 3, 13, 23,
                   100, 110, 120, 101, 111, 121, 102, 112, 122, 103, 113, 123};
	for (int i = 0; i < 24; ++i) {
		ASSERT_EQ(M[i], expected[i]);
	}
}

TEST_F(MatrixTest, check_iterator_standard)
{
    std::vector<int> expected = {0, 10, 20, 1, 11, 21, 2, 12, 22, 3, 13, 23,
                   100, 110, 120, 101, 111, 121, 102, 112, 122, 103, 113, 123};
    auto itm = M.begin();
    auto ite = expected.begin();
	for (; itm != M.end(); ++itm, ++ite) {
		ASSERT_EQ(*itm, *ite);
	}
}

TEST_F(MatrixTest, check_get_standard)
{
	for (int c = 0; c < 4; ++c) {
	    for (int r = 0; r < 3; ++r) {
	        for (int s = 0; s < 2; ++s) {
		        ASSERT_EQ(M.get(r, c, s), s*100 + r*10 + c);
		    }
		}
	}
}

TEST_F(MatrixTest, check_indexing_operator_transposed)
{
    Matrix n = M.T();
    std::vector<int> expected = {0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23,
                   100, 101, 102, 103, 110, 111, 112, 113, 120, 121, 122, 123,};
	for (int i = 0; i < 24; ++i) {
		ASSERT_EQ(n[i], expected[i]);
	}
}

TEST_F(MatrixTest, check_iterator_transposed)
{
    std::vector<int> expected = {0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23,
                       100, 101, 102, 103, 110, 111, 112, 113, 120, 121, 122, 123,};
    Matrix n = M.T();
    auto itm = n.begin();
    auto ite = expected.begin();
	for (; itm != n.end(); ++itm, ++ite) {
		ASSERT_EQ(*itm, *ite);
	}
}


TEST_F(MatrixTest, check_get_transposed)
{
    Matrix n = M.T();
	for (int c = 0; c < 3; ++c) {
	    for (int r = 0; r < 4; ++r) {
	        for (int s = 0; s < 2; ++s) {
		        ASSERT_EQ(n.get(r, c, s), s*100 + c*10 + r);
		    }
		}
	}
}

TEST_F(MatrixTest, check_indexing_time_slice)
{
    Matrix n = M.slice(1);
    ASSERT_EQ(n.n_columns, 4);
    ASSERT_EQ(n.n_rows, 3);
    ASSERT_EQ(n.n_slices, 1);
    std::vector<int> expected = {100, 110, 120, 101, 111, 121, 102, 112, 122, 103, 113, 123};
	for (int i = 0; i < 12; ++i) {
		ASSERT_EQ(n[i], expected[i]);
	}
}

TEST_F(MatrixTest, check_iterator_time_slice)
{
    Matrix n = M.slice(1);
    std::vector<int> expected = {100, 110, 120, 101, 111, 121, 102, 112, 122, 103, 113, 123};
    auto itm = n.begin();
    auto ite = expected.begin();
	for (; itm != n.end(); ++itm, ++ite) {
		ASSERT_EQ(*itm, *ite);
	}
}

TEST_F(MatrixTest, check_get_time_slice)
{
    Matrix n = M.slice(1);
    ASSERT_EQ(n.n_columns, 4);
    ASSERT_EQ(n.n_rows, 3);
    ASSERT_EQ(n.n_slices, 1);
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 3; ++r) {
            ASSERT_EQ(n.get(r, c, 0), 100 + r*10 + c);
        }
    }
}

TEST_F(MatrixTest, check_indexing_time_slice_transposed)
{
    Matrix n = M.slice(1).T();
    ASSERT_EQ(n.n_columns, 3);
    ASSERT_EQ(n.n_rows, 4);
    ASSERT_EQ(n.n_slices, 1);
    std::vector<int> expected = {100, 101, 102, 103, 110, 111, 112, 113, 120, 121, 122, 123};
	for (int i = 0; i < 12; ++i) {
		ASSERT_EQ(n[i], expected[i]);
	}
}

TEST_F(MatrixTest, check_iterator_time_slice_transposed)
{
    Matrix n = M.slice(1).T();
    ASSERT_EQ(n.n_columns, 3);
    ASSERT_EQ(n.n_rows, 4);
    ASSERT_EQ(n.n_slices, 1);
    std::vector<int> expected = {100, 101, 102, 103, 110, 111, 112, 113, 120, 121, 122, 123};
    auto itm = n.begin();
    auto ite = expected.begin();
	for (; itm != n.end(); ++itm, ++ite) {
		ASSERT_EQ(*itm, *ite);
	}
}

TEST_F(MatrixTest, check_get_time_slice_transposed)
{
    Matrix n = M.slice(1).T();
    ASSERT_EQ(n.n_columns, 3);
    ASSERT_EQ(n.n_rows, 4);
    ASSERT_EQ(n.n_slices, 1);
    for (int c = 0; c < 3; ++c) {
        for (int r = 0; r < 4; ++r) {
            ASSERT_EQ(n.get(r, c, 0), 100 + r + c*10);
        }
    }
}


TEST_F(MatrixTest, check_get_row_slice)
{
    Matrix n = M.row_slice(1);
    ASSERT_EQ(n.n_columns, 4);
    ASSERT_EQ(n.n_rows, 1);
    ASSERT_EQ(n.n_slices, 2);
    for (int c = 0; c < 4; ++c) {
        for (int s = 0; s < 2; ++s) {
            ASSERT_EQ(n.get(0, c, s), s*100 + 10 + c);
        }
    }
}

TEST_F(MatrixTest, check_indexing_row_slice)
{
    Matrix n = M.row_slice(1);
    ASSERT_EQ(n.n_columns, 4);
    ASSERT_EQ(n.n_rows, 1);
    ASSERT_EQ(n.n_slices, 2);
    std::vector<int> expected = {10, 11, 12, 13, 110, 111, 112, 113};
    for (int i = 0; i < 8; ++i) {
    	ASSERT_EQ(n[i], expected[i]);
    }
}

TEST_F(MatrixTest, check_iterator_row_slice)
{
    Matrix n = M.row_slice(1);
        ASSERT_EQ(n.n_columns, 4);
        ASSERT_EQ(n.n_rows, 1);
        ASSERT_EQ(n.n_slices, 2);
        std::vector<int> expected = {10, 11, 12, 13, 110, 111, 112, 113};
    auto itm = n.begin();
    auto ite = expected.begin();
	for (; itm != n.end(); ++itm, ++ite) {
		ASSERT_EQ(*itm, *ite);
	}
}

TEST_F(MatrixTest, check_get_row_slice_transposed)
{
    Matrix n = M.row_slice(1).T();
    ASSERT_EQ(n.n_columns, 1);
    ASSERT_EQ(n.n_rows, 4);
    ASSERT_EQ(n.n_slices, 2);
    for (int r = 0; r < 4; ++r) {
        for (int s = 0; s < 2; ++s) {
            ASSERT_EQ(n.get(r, 0, s), s*100 + 10 + r);
        }
    }
}

TEST_F(MatrixTest, check_indexing_row_slice_transposed)
{
    Matrix n = M.row_slice(1).T();
    ASSERT_EQ(n.n_columns, 1);
    ASSERT_EQ(n.n_rows, 4);
    ASSERT_EQ(n.n_slices, 2);
    std::vector<int> expected = {10, 11, 12, 13, 110, 111, 112, 113};
    for (int i = 0; i < 8; ++i) {
    	ASSERT_EQ(n[i], expected[i]);
    }
}

TEST_F(MatrixTest, check_iterator_row_slice_transposed)
{
    Matrix n = M.row_slice(1).T();
    ASSERT_EQ(n.n_columns, 1);
    ASSERT_EQ(n.n_rows, 4);
    ASSERT_EQ(n.n_slices, 2);
    std::vector<int> expected = {10, 11, 12, 13, 110, 111, 112, 113};
    auto itm = n.begin();
    auto ite = expected.begin();
	for (; itm != n.end(); ++itm, ++ite) {
		ASSERT_EQ(*itm, *ite);
	}
}


TEST_F(MatrixTest, check_get_multirow_slice)
{
    Matrix n = M.row_slice(1,3);
    ASSERT_EQ(n.n_columns, 4);
    ASSERT_EQ(n.n_rows, 2);
    ASSERT_EQ(n.n_slices, 2);
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 2; ++r) {
            for (int s = 0; s < 2; ++s) {
                ASSERT_EQ(n.get(r, c, s), s*100 + (r+1)*10 + c);
            }
        }
    }
}

TEST_F(MatrixTest, check_indexing_multirow_slice)
{
    Matrix n = M.row_slice(1,3);
    ASSERT_EQ(n.n_columns, 4);
    ASSERT_EQ(n.n_rows, 2);
    ASSERT_EQ(n.n_slices, 2);
    std::vector<int> expected = {10, 20, 11, 21, 12, 22, 13, 23, 110, 120, 111, 121, 112, 122, 113, 123};
    for (int i = 0; i < 8; ++i) {
    	ASSERT_EQ(n[i], expected[i]);
    }
}

TEST_F(MatrixTest, check_iterator_multirow_slice)
{
    Matrix n = M.row_slice(1,3);
    ASSERT_EQ(n.n_columns, 4);
    ASSERT_EQ(n.n_rows, 2);
    ASSERT_EQ(n.n_slices, 2);
    std::vector<int> expected = {10, 20, 11, 21, 12, 22, 13, 23, 110, 120, 111, 121, 112, 122, 113, 123};
    auto itm = n.begin();
    auto ite = expected.begin();
	for (; itm != n.end(); ++itm, ++ite) {
		ASSERT_EQ(*itm, *ite);
	}
}


TEST_F(MatrixTest, check_get_multirow_slice_transposed)
{
    Matrix n = M.row_slice(1,3).T();
    ASSERT_EQ(n.n_columns, 2);
    ASSERT_EQ(n.n_rows, 4);
    ASSERT_EQ(n.n_slices, 2);
    for (int c = 0; c < 2; ++c) {
        for (int r = 0; r < 4; ++r) {
            for (int s = 0; s < 2; ++s) {
                ASSERT_EQ(n.get(r, c, s), s*100 + (c+1)*10 + r);
            }
        }
    }
}

TEST_F(MatrixTest, check_indexing_multirow_slice_transposed)
{
    Matrix n = M.row_slice(1,3).T();
    ASSERT_EQ(n.n_columns, 2);
    ASSERT_EQ(n.n_rows, 4);
    ASSERT_EQ(n.n_slices, 2);
    std::vector<int> expected = {10, 11, 12, 13, 20, 21, 22, 23, 110, 111, 112, 113, 120, 121, 122, 123};
    for (int i = 0; i < 8; ++i) {
    	ASSERT_EQ(n[i], expected[i]);
    }
}

TEST_F(MatrixTest, check_iterator_multirow_slice_transposed)
{
    Matrix n = M.row_slice(1,3).T();
    ASSERT_EQ(n.n_columns, 2);
    ASSERT_EQ(n.n_rows, 4);
    ASSERT_EQ(n.n_slices, 2);
    std::vector<int> expected = {10, 11, 12, 13, 20, 21, 22, 23, 110, 111, 112, 113, 120, 121, 122, 123};
    auto itm = n.begin();
    auto ite = expected.begin();
	for (; itm != n.end(); ++itm, ++ite) {
		ASSERT_EQ(*itm, *ite);
	}
}

TEST_F(MatrixTest, check_set_all_elements_to_one)
{
    M.set_all_elements_to(1.0);
    for (d_type& m : M) {
		ASSERT_EQ(m, 1.0);
	}
}

TEST_F(MatrixTest, check_set_all_elements_to_zero)
{
    M.set_all_elements_to(0.0);
    for (d_type& m : M) {
		ASSERT_EQ(m, 0.0);
	}
}

TEST_F(MatrixTest, check_set_some_elements_to_zero)
{
    Matrix o = M.slice(0);
    Matrix n = M.slice(1);
    o.set_all_elements_to(0.0);
    for (d_type& m : o) {
		ASSERT_EQ(m, 0.0);
	}
	for (d_type& m : n) {
	    ASSERT_NE(m, 0.0);
	}
}