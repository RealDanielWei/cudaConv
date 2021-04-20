
set(LIB_NAME "cudaLib")

add_library(${LIB_NAME} cudaLib.cu cudaLib.h)

target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
