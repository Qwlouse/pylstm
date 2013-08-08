# The layers library
ADD_LIBRARY(layers SHARED
            fwd_layer.cpp
            rev_layer.cpp
            lstm_layer.cpp
            lstm97_layer.cpp
            mrnn_layer.cpp
            rnn_layer.cpp
            arnn_layer.cpp
            )

TARGET_LINK_LIBRARIES(layers ${BLAS_LIBRARIES} core matrix)

IF(LOGGING_ENABLED)
    TARGET_LINK_LIBRARIES(layers ${LOG4CXX_LIBRARIES})
ENDIF(LOGGING_ENABLED)
