#include "mpi.h"
#include "cuda_runtime.h"
#include <iostream>

int main(int argc, char** argv) {

    size_t elem_num = 4096;
    int tag1 = 10;
	int tag2 = 20;
    
    int nprocs, rank, local_rank, partner_rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &shmcomm);
    MPI_Comm_rank(shmcomm, &local_rank);

    int *loc_ranks = (int*) malloc(nprocs*sizeof(int));
    MPI_Allgather(&local_rank, 1, MPI_INT, loc_ranks, 1, MPI_INT, MPI_COMM_WORLD);
    for(int i=0; i<nprocs; i++) {
        if((i != rank) && (loc_ranks[i] == loc_ranks[rank])) {
            partner_rank = i;
            break;
        }
    }
    free(loc_ranks);

    std::cout<< "Start MPI Allgather between rank " << rank << " <--> " << partner_rank << std::endl;

    double *host_buf;

    host_buf = (double*) malloc(elem_num*sizeof(double));
    for(int i=0; i<elem_num; i++) {
        host_buf[i] = 1.0;
    }

    int min_rank, max_rank;
    min_rank = rank < partner_rank ? rank : partner_rank;
    max_rank = rank > partner_rank ? rank : partner_rank;

    while(1) {
        if(rank == min_rank){
            MPI_Send(host_buf, elem_num, MPI_DOUBLE, partner_rank, tag1, MPI_COMM_WORLD);
            MPI_Recv(host_buf, elem_num, MPI_DOUBLE, partner_rank, tag2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else if(rank == max_rank){
            MPI_Recv(host_buf, elem_num, MPI_DOUBLE, partner_rank, tag1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(host_buf, elem_num, MPI_DOUBLE, partner_rank, tag2, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}