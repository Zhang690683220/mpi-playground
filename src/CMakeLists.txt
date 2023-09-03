add_compile_options(-target-accel=nvidia80)
add_subdirectory(allgather)
add_subdirectory(cudamemcpy)
add_subdirectory(pingpong)