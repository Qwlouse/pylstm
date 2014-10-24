# The layers library
ADD_LIBRARY(layers SHARED
            forward_layer.cpp
            reverse_layer.cpp
            lstm_layer.cpp
            static_lstm_layer.cpp
            lstm97_layer.cpp
            mrnn_layer.cpp
            rnn_layer.cpp
            clockwork_layer.cpp
            ctc.cpp
            dropout_layer.cpp
            hf_final_layer.cpp
            lwta_layer.cpp
            )

TARGET_LINK_LIBRARIES(layers ${BLAS_LIBRARIES} core matrix)

