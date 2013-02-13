# The layers library
ADD_LIBRARY(layers SHARED
            lstm_layer.cpp
            fwd_layer.cpp
            )

TARGET_LINK_LIBRARIES(layers ${MATLAB_LIBRARIES} core matrix)

IF(LOGGING_ENABLED)
    TARGET_LINK_LIBRARIES(layers ${LOG4CXX_LIBRARIES})
ENDIF(LOGGING_ENABLED)

ADD_EXECUTABLE(test_layers test.cpp)
TARGET_LINK_LIBRARIES(test_layers layers)
