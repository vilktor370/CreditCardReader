cmake_minimum_required(VERSION 3.0)
project(main)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

set(HEADERS util/Common.h)
set(SOURCES util/Common.cpp main.cpp)
# Add source files
add_executable(main ${SOURCES} ${HEADERS})

# Link OpenCV libraries
target_link_libraries(main ${OpenCV_LIBS})
