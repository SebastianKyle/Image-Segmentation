cmake_minimum_required(VERSION 3.6.0)
project(ImageSegmentation VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(OpenMP)

include_directories(
    $(CMAKE_SOURCE_DIR)/header
    $(CMAKE_SOURCE_DIR)/src
)

set(SOURCES
    main.cpp
    src/libs.cpp
    src/MorphologicalProcessing.cpp
    src/ImageSegmentation.cpp
)

add_executable(ImageSegmentation ${SOURCES})

target_link_libraries( ImageSegmentation ${OpenCV_LIBS} OpenMP::OpenMP_CXX )

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
