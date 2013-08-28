# The core library
ADD_LIBRARY(core STATIC
            Exceptions.cpp
            Assert.cpp)

TARGET_LINK_LIBRARIES(core dl)

