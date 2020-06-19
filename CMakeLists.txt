cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(CUDA_EVENT_BENCH LANGUAGES C CXX CUDA)

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-error=deprecated-declarations -static-libgcc -static-libstdc++")

    option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" ON)
    if(CMAKE_CXX11_ABI)
        message(STATUS "Enabling the GLIBCXX11 ABI")
    else()
        message(STATUS "Disabling the GLIBCXX11 ABI")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0")
    endif(CMAKE_CXX11_ABI)
endif(CMAKE_COMPILER_IS_GNUCXX)

option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)

if(CUDA_STATIC_RUNTIME)
    message(STATUS "Enabling static linking of cudart")
    set(CUDART_LIBRARY "cudart_static")
else()
    set(CUDART_LIBRARY "cudart")
endif(CUDA_STATIC_RUNTIME)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})
include(ConfigureGoogleBenchmark)

include_directories("${GBENCH_INCLUDE_DIR}" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
link_directories("${GBENCH_LIBRARY_DIR}" "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")

add_executable(cuda_event_bench event_bench.cpp)
#set_target_properties(cuda_event_bench PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cuda_event_bench PRIVATE benchmark pthread "${CUDART_LIBRARY}" dl rt)