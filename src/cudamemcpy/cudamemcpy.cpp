#include "CLI11.hpp"
#include "timer.hpp"
#include "mpi.h"
#include "cuda_runtime.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    CLI::App app{"CUDA D->H Memcpy"};
    int niter = 10;
    std::string msg_size;
    int shift;
    app.add_option("-i, --iter", niter, "Total iterations", true);
    app.add_option("-s, --size", msg_size, "Data size that to be copied from device to host. For example"
                    "32K ,64M ...")->required();

    CLI11_PARSE(app, argc, argv);

    if(msg_size.back() == 'k' || msg_size.back() == 'K') {
        shift = 10;
    } else if(msg_size.back() == 'm' || msg_size.back() == 'M') {
        shift = 20;
    } else if(msg_size.back() == 'g' || msg_size.back() == 'G') {
        shift = 30;
    }

    int nprocs, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0) {
        std::cout<< "Start CUDA Memcpy for " << msg_size <<" messsage on " << nprocs
                 << " processes." << std::endl;
    }

    msg_size.pop_back();
    unsigned long long data_size = std::stoull(msg_size);
    data_size = data_size << shift;

    double *send_buf, *recv_buf, *host_buf;

    size_t elem_num = data_size / sizeof(double);

    host_buf = (double*) malloc(elem_num*sizeof(double));
    for(int i=0; i<elem_num; i++) {
        host_buf[i] = 1.0;
    }

    cudaMalloc((void**)&send_buf, elem_num*sizeof(double));
    cudaMalloc((void**)&recv_buf, elem_num*nprocs*sizeof(double));
    cudaMemcpy(send_buf, host_buf, elem_num*sizeof(double), cudaMemcpyHostToDevice);
    free(host_buf);

    Timer timer_memcpy;

    for(int it; it<niter; it++) {
        MPI_Barrier(MPI_COMM_WORLD);
        timer_memcpy.start();
        cudaMemcpy(recv_buf, send_buf, elem_num*sizeof(double), cudaMemcpyDeviceToHost);
        double time = timer_memcpy.stop();
        std::cout << "Iter: " << it << " Rank: " << rank << "cudaMemcpy time = " << time << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    cudaFree(send_buf);
    cudaFree(recv_buf);

    MPI_Finalize();

    return 0;

}