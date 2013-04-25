# The matrix library
ADD_LIBRARY(matrix SHARED
            matrix.cpp
            matrix_operation.cpp
            matrix_container.cpp)

TARGET_LINK_LIBRARIES(matrix ${MATLAB_LIBRARIES} core)

IF(LOGGING_ENABLED)
    TARGET_LINK_LIBRARIES(matrix ${LOG4CXX_LIBRARIES})
ENDIF(LOGGING_ENABLED)
