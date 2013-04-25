# The layers library
ADD_LIBRARY(layers SHARED
            fwd_layer.cpp
            rev_layer.cpp
            lstm_layer.cpp
            rnn_layer.cpp
            view_container.cpp
            )

TARGET_LINK_LIBRARIES(layers ${MATLAB_LIBRARIES} core matrix)

IF(LOGGING_ENABLED)
    TARGET_LINK_LIBRARIES(layers ${LOG4CXX_LIBRARIES})
ENDIF(LOGGING_ENABLED)
