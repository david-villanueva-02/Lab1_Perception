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
project(lab1 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")

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

# Calibrate
add_executable(calibrate
    src/calibrate.cpp
)

target_include_directories(calibrate PUBLIC
${OpenCV_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(calibrate PRIVATE
${OpenCV_LIBS}
)

target_compile_options(calibrate PRIVATE -O3)

# Pose estimation
add_executable(pose_estimation
    src/pose_estimation.cpp
)

target_include_directories(pose_estimation PUBLIC
${OpenCV_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(pose_estimation PRIVATE
${OpenCV_LIBS}
)

target_compile_options(pose_estimation PRIVATE -O3)

# Draw cube
add_executable(draw_cube
    src/draw_cube.cpp
)

target_include_directories(draw_cube PUBLIC
${OpenCV_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(draw_cube PRIVATE
${OpenCV_LIBS}
)

target_compile_options(draw_cube PRIVATE -O3)

# Relative pose estimation
add_executable(relative_pose_estimation
    src/relative_pose_estimation.cpp
)

target_include_directories(relative_pose_estimation PUBLIC
${OpenCV_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(relative_pose_estimation PRIVATE
${OpenCV_LIBS}
)

target_compile_options(relative_pose_estimation PRIVATE -O3)

# Distance estimation
add_executable(distance_estimation
    src/distance_estimation.cpp
)

target_include_directories(distance_estimation PUBLIC
${OpenCV_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(distance_estimation PRIVATE
${OpenCV_LIBS}
)

target_compile_options(distance_estimation PRIVATE -O3)