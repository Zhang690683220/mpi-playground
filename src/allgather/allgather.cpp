#include "CLI11.hpp"
#include "mpi.h"
#include "cuda_runtime.h"
#include <iostream>

int main(int argc, char** argv) {
    CLI::App app{"MPI Allgather"};

    bool cpu = false;
    bool gpu = false;
    size_t elem_num = 4096;

    app.add_flag("--cpu", cpu, "MPI Allgather to/from CPU buffer");
    app.add_flag("--gpu", gpu, "MPI Allgather to/from GPU buffer");

    CLI11_PARSE(app, argc, argv);

    if((cpu && gpu) || (!cpu && !gpu)) {
        std::cout<< "Please specify either --cpu or --gpu flag"<< std::endl;
        return -1;
    }

    int nprocs, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout<< "Start MPI Allgather between " << nprocs << " processes." << std::endl;

    double *send_buf, *recv_buf, *host_buf;

    host_buf = (double*) malloc(elem_num*sizeof(double));
    for(int i=0; i<elem_num; i++) {
        host_buf[i] = 1.0;
    }

    if(gpu) {
        cudaMalloc((void**)&send_buf, elem_num*sizeof(double));
        cudaMalloc((void**)&recv_buf, elem_num*nprocs*sizeof(double));
        cudaMemcpy(send_buf, host_buf, elem_num*sizeof(double), cudaMemcpyHostToDevice);
        free(host_buf);
    } else if(cpu) {
        send_buf = host_buf;
        recv_buf = (double*) malloc(elem_num*nprocs*sizeof(double));
    }

    while(1) {
        MPI_Allgather(send_buf, elem_num, MPI_DOUBLE, recv_buf, elem_num, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}