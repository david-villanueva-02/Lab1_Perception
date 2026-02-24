# cmake_minimum_required(VERSION 3.1)
# project(example)

# set (CMAKE_CXX_STANDARD 11)
# find_package(OpenCV REQUIRED)

# include_directories(${OPENCV_INCLUDE_DIRS})
# include_directories(${PROJECT_SOURCE_DIR}/include)

# link_directories(${OpenCV_LIBRARY_DIRS})

# set(example_src
# src/main.cpp
# )

# add_executable(example ${example_src})
#     target_link_libraries(example
#     ${OpenCV_LIBRARIES}
# )

# target_compile_options(example
#     PRIVATE -O3 -std=c++11
# )

cmake_minimum_required(VERSION 3.16)
project(example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# Example code
add_executable(example
src/main.cpp
)

target_include_directories(example PRIVATE
${OpenCV_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(example PRIVATE
${OpenCV_LIBS}
)

target_compile_options(example PRIVATE -O3)

# Generate marker
add_executable(generate_marker
src/generate_marker.cpp
)

target_include_directories(generate_marker PRIVATE
${OpenCV_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(generate_marker PRIVATE
${OpenCV_LIBS}
)

target_compile_options(generate_marker PRIVATE -O3)

# Generate board
add_executable(generate_board
src/generate_board.cpp
)

target_include_directories(generate_board PRIVATE
${OpenCV_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(generate_board PRIVATE
${OpenCV_LIBS}
)

target_compile_options(generate_board PRIVATE -O3)

# Detect marker
add_executable(detection
    src/detection.cpp
)

target_include_directories(detection PRIVATE
${OpenCV_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(detection PRIVATE
${OpenCV_LIBS}
)

target_compile_options(detection PRIVATE -O3)