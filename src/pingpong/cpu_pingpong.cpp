#include "mpi.h"
#include "cuda_runtime.h"
#include <iostream>

int main(int argc, char** argv) {

    size_t elem_num = 4096;
    int tag1 = 10;
	int tag2 = 20;
    
    int nprocs, rank, partner_rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout<< "Start MPI Allgather between 2 processes." << std::endl;

    double *host_buf;

    host_buf = (double*) malloc(elem_num*sizeof(double));
    for(int i=0; i<elem_num; i++) {
        host_buf[i] = 1.0;
    }

    while(1) {
        if(rank == 0){
            MPI_Send(host_buf, elem_num, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
            MPI_Recv(host_buf, elem_num, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else if(rank == 1){
            MPI_Recv(host_buf, elem_num, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(host_buf, elem_num, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}