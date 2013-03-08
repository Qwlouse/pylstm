# The layers library
ADD_LIBRARY(layers SHARED
            fwd_layer.cpp
            )

TARGET_LINK_LIBRARIES(layers ${MATLAB_LIBRARIES} core matrix)

IF(LOGGING_ENABLED)
    TARGET_LINK_LIBRARIES(layers ${LOG4CXX_LIBRARIES})
ENDIF(LOGGING_ENABLED)
