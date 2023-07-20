#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <assert.h>
#include <malloc.h>
#include <mpi.h>
#include <cuda_runtime.h>


cudaError_t init_bound(double * arr_d, double* arr2_d, int rank, int size_rank, double gradient, int start, int end, int size_x, int size_y);


int main(int argc, char* argv[]) 
{
    assert(argc == 4);

    printf("START PROGRAM!\n");

	int size = atoi(argv[1]);
	double accur = atof(argv[2]);
	int iter_max = atoi(argv[3]);

    int mpi_er;

    mpi_er = MPI_Init(&argc, &argv);
    assert(MPI_SUCCESS == mpi_er);
    printf("------INIT MPI SUCCESS!\n");

    int rank, size_rank, num_devices = 0;

    cudaError_t cuda_er;

    mpi_er = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    assert(MPI_SUCCESS == mpi_er);
    mpi_er = MPI_Comm_size(MPI_COMM_WORLD, &size_rank);
    assert(MPI_SUCCESS == mpi_er);
    printf("------INIT MPI_RANKs AND TOTAL_RANK_SIZE SUCCESS!\n");

    cuda_er=cudaGetDeviceCount(&num_devices);
    assert(cuda_er == 0);
	printf("------COUNTED DEVICES SUCCESS!!! - TOTAL = %d gpu's\n", num_devices);

    int bias_1 = 2;
    double gradient = 10.0 / size;

    double *arr,* arr2;
	// размер общий нашего тензора, здесь тензор для удобства и скорости представляется в виде сложенного вектора
	size_t gl_size = (size + bias_1)*(size + bias_1)*sizeof(double);
    arr = (double*)calloc((size + bias_1)*(size + bias_1),sizeof(double));
    arr2 = (double*)calloc((size + bias_1)*(size + bias_1),sizeof(double));

    int loc_y = (size+bias_1)/size_rank;
    int mod_y = (size+bias_1)%size_rank;
    
    int loc2_y = loc_y;

    double *arr_d, *arr2_d;

    if ((mod_y != 0) && (rank == size_rank-1))
    {
        loc_y += mod_y;
    } 

    int bias_halo = (rank == 0) ? 1 : 2;
   
    cuda_er = cudaMalloc(&arr_d, (size + bias_1) * (loc_y + bias_halo) * sizeof(double));
    assert(cuda_er == 0);
    printf("------ALLOCATE MEMORY (arr_d) SUCCESS!!!\n");

    cuda_er = cudaMalloc(&arr2_d, (size + bias_1) * (loc_y + bias_halo) * sizeof(double));
    assert(cuda_er == 0);
    printf("------ALLOCATE MEMORY (arr2_d) SUCCESS!!!\n");
    
    cuda_er = cudaMemset(arr_d, 0, (size + bias_1) * (loc_y + bias_halo) * sizeof(double));
    assert(cuda_er == 0);
    cuda_er = cudaMemset(arr2_d, 0, (size + bias_1) * (loc_y + bias_halo) * sizeof(double));
    assert(cuda_er == 0);
    
    double *arr_debug = (double*)calloc((size + bias_1) * (loc_y + bias_halo), sizeof(double));

    // далее отметим границы начала и конца наших локальных массивов

    int start_arr, end_arr;

    start_arr = 0;
    end_arr = start_arr + ((size + bias_1) * (loc_y + bias_halo))-1;

    cuda_er = init_bound(arr_d,arr2_d, rank, size_rank, gradient, start_arr, end_arr, (size + bias_1), loc2_y);

    cuda_er = cudaMemcpy(arr_debug, arr_d, (size + bias_1) * (loc_y + bias_halo) * sizeof(double), cudaMemcpyDeviceToHost);
    assert(cuda_er == 0);
    printf("------COPY MEMORY VRAM -> RAM (arr_d -> arr_debug) SUCCESS!!!\n");

    if (rank == 1)
    {
        int k=0;
        for (int i=0;i<(size + bias_1) * (loc_y + bias_halo);i++)
        {
            if (k==(size + bias_1))
            {
                printf("\n");
                k = 0;
            }
            k = k +1;
            //if (i == start_arr) arr_debug[i] = 1.0;
            //if (i == end_arr) arr_debug[i] = 2.0;
            printf("%d ", (int)arr_debug[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    printf("PROGRAM HAS ENDED SUCCESSFULLY!\n");
    return 0;
}