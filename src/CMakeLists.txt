
add_executable(${PROJECT_NAME} main.cpp)

if(USE_CUDA)
	target_include_directories(${PROJECT_NAME} PUBLIC ${ROOT_PATH}/lib/cudaLib)

	target_link_directories(${PROJECT_NAME} PUBLIC ${ROOT_PATH}/lib/cudaLib)

	target_link_libraries(${PROJECT_NAME} PUBLIC cudaLib)
endif()
