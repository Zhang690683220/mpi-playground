add_executable(cpu_pingpong cpu_pingpong.cpp)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
    target_include_directories(cpu_pingpong PRIVATE ${INCLUDE_PATH} ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(cpu_pingpong PUBLIC MPI::MPI_CXX
                            CUDA::cudart CUDA::cuda_driver)
else()
    target_include_directories(cpu_pingpong PRIVATE ${INCLUDE_PATH} ${CUDA_INCLUDE_DIRS})
    target_link_libraries(cpu_pingpong PUBLIC MPI::MPI_CXX
                            ${CUDA_LIBRARIES})
endif()