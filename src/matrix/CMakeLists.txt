# The matrix library
ADD_LIBRARY(matrix SHARED
            matrix.cpp
            matrix_cpu.cpp
            matrix_operation_cpu.cpp)

TARGET_LINK_LIBRARIES(matrix ${MATLAB_LIBRARIES})

IF(LOGGING_ENABLED)
    TARGET_LINK_LIBRARIES(matrix ${LOG4CXX_LIBRARIES})
ENDIF(LOGGING_ENABLED)

ADD_EXECUTABLE(test_matrix test.cpp)
TARGET_LINK_LIBRARIES(test_matrix matrix)
