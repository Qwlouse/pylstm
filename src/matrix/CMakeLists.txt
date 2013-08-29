# The matrix library
ADD_LIBRARY(matrix SHARED
            matrix.cpp
            matrix_operation.cpp
            matrix_container.cpp)

TARGET_LINK_LIBRARIES(matrix ${BLAS_LIBRARIES} core)

