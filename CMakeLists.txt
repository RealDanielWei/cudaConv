cmake_minimum_required(VERSION 3.18.1)

project(cudaConv)

set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

option(ENABLE_CUDA "Enable CUDA" ON)

if (ENABLE_CUDA)
	include(CheckLanguage)
	check_language(CUDA)
	if(CMAKE_CUDA_COMPILER)
		enable_language(CUDA)
		set(USE_CUDA ON)
		add_definitions(-DUSE_CUDA)
		add_subdirectory(lib/cudaLib)
		message(STATUS "Use CUDA!")
	else()
		message(STATUS "Cannot Find CUDA!")
	endif()
endif()

add_subdirectory(src)
