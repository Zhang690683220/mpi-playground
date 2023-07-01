add_executable(allgather allgather.cpp)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
    target_include_directories(allgather PRIVATE ${INCLUDE_PATH} ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(allgather PUBLIC MPI::MPI_CXX
                            CUDA::cudart CUDA::cuda_driver)
else()
    target_include_directories(allgather PRIVATE ${INCLUDE_PATH} ${CUDA_INCLUDE_DIRS})
    target_link_libraries(allgather PUBLIC MPI::MPI_CXX
                            ${CUDA_LIBRARIES})
endif()