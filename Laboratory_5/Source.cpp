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


cudaError_t init_bound(double * main_dev, double * arr_d, double gradient,  int size_x, int size_rank, int rank, int start, int end);

void ComputeTensor(double* arr_d,double* arr2_d, int size_x, int start, int end, int rank, int size_rank, cudaStream_t stream);

void difference(double* arr_dif,double* arr_d,double* arr2_d,int start,int end,cudaStream_t stream, double* out);

void cub_cmp(double* arr_dif,double* arr_d,double* arr2_d,int start,int end,cudaStream_t stream, double* out);

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

	// размер общий нашего тензора, здесь тензор для удобства и скорости представляется в виде сложенного вектора
	size_t gl_size = (size + bias_1)*(size + bias_1)*sizeof(double);

    double *main_dev;

    cuda_er = cudaMalloc(&main_dev, gl_size);
    assert(cuda_er == 0);
    printf("------ALLOCATE MEMORY (arr_d) SUCCESS!!!\n");
 
    cuda_er = cudaMemset(main_dev, 0, gl_size);
    assert(cuda_er == 0);

    // далее отметим границы начала и конца наших локальных массивов

    int loc_y = (size+bias_1)/size_rank;
    int mod_y = (size+bias_1)%size_rank;
    
    int loc2_y = loc_y;

    double *arr_d, *arr2_d, *arr_dif, *arr_difh;

    if ((mod_y != 0) && (rank == size_rank-1))
    {
        loc_y += mod_y;
    } 

    int bias_halo = 2;

    cuda_er = cudaMalloc(&arr_d, (size + bias_1) * (loc_y + bias_halo) * sizeof(double));
    assert(cuda_er == 0);
    cuda_er = cudaMemset(arr_d, 0, (size + bias_1) * (loc_y + bias_halo) * sizeof(double));
    assert(cuda_er == 0);
    printf("------ALLOCATE MEMORY (arr_d) SUCCESS!!!\n");

    cuda_er = cudaMalloc(&arr2_d, (size + bias_1) * (loc_y + bias_halo) * sizeof(double));
    assert(cuda_er == 0);
    cuda_er = cudaMemset(arr2_d, 0, (size + bias_1) * (loc_y + bias_halo) * sizeof(double));
    assert(cuda_er == 0);
    printf("------ALLOCATE MEMORY (arr_d) SUCCESS!!!\n");

    double* out_max_data;

	cuda_er = cudaMalloc(&out_max_data, sizeof(double));
	assert(cuda_er == 0);
	printf("------ALLOCATED MEMORY REDUCE_BUFFER SUCCESS!!!\n");

    cuda_er = cudaMalloc(&arr_dif,(size + bias_1) * (loc_y + bias_halo) * sizeof(double));
    assert(cuda_er == 0);

    //int part = (size + bias_1) * (loc2_y + mod_y);
    //int full_part = ((size + bias_1) * (loc2_y + mod_y))*size_rank;

    arr_difh = (double*)malloc((size + bias_1) * (loc_y + bias_halo) * sizeof(double));

    double *arr_debug = (double*)malloc((size + bias_1) * (loc_y + bias_halo) * sizeof(double));

    printf("INDIVIDUAL FOR '0' RANK...!!!\n");
    //double * arr_table = (double*)malloc(full_part * sizeof(double));

    int start_arr = 0;
    int end_arr = start_arr + ((size + bias_1) * (loc_y + bias_halo))-1;

    cuda_er = init_bound(main_dev, arr_d, gradient, size+bias_1, size_rank, rank, start_arr, end_arr);

    cuda_er = cudaMemcpy(arr2_d, arr_d, (size + bias_1) * (loc_y + bias_halo) * sizeof(double), cudaMemcpyDeviceToDevice);
    assert(cuda_er == 0);
    printf("------COPY MEMORY VRAM -> VRAM (arr_d -> arr2_d) SUCCESS!!!\n");

    //MPI_Status stat;
    //MPI_Request req;

    int prev = rank - 1;
    int node = rank;
    int next = rank + 1;
    if (node == 0) prev = size_rank-1;
    if (node == size_rank-1) next = 0;
    
    // mpi_er = MPI_Sendrecv(arr2_d + end_arr - ((size+bias_1)*2)+1, (size+bias_1), MPI_DOUBLE, next, 0,
    //                         arr2_d, (size+bias_1), MPI_DOUBLE, prev, 0, MPI_COMM_WORLD,
    //                         MPI_STATUS_IGNORE);
    
    // mpi_er = MPI_Sendrecv(arr2_d + (size+bias_1), (size+bias_1), MPI_DOUBLE, prev, 0,
    //                         arr2_d + end_arr - (size+bias_1)+1, (size+bias_1), MPI_DOUBLE, next, 0, MPI_COMM_WORLD,
    //                         MPI_STATUS_IGNORE);

    cuda_er = cudaDeviceSynchronize();
    assert(cuda_er == 0);

    double error = 1;
    int iter = 0;
    double *ptr = NULL;
    cuda_er = cudaMalloc(&ptr,sizeof(double));
    assert(cuda_er == 0);

    cudaStream_t stream;
    cudaEvent_t compute_ok;
    cuda_er = cudaStreamCreate(&stream);
    cuda_er = cudaEventCreateWithFlags(&compute_ok, cudaEventDisableTiming);
    assert(cuda_er == 0);

    MPI_Barrier(MPI_COMM_WORLD);

    while ((iter < 1000) && (error > accur))
    {
        iter++;
        cuda_er = cudaMemsetAsync(arr_dif, 0,(size + bias_1) * (loc_y + bias_halo) * sizeof(double), stream);
        assert(cuda_er == 0);

        ComputeTensor(arr_d, arr2_d, (size+bias_1), start_arr, end_arr, rank, size_rank, stream);
        cuda_er = cudaEventRecord(compute_ok, stream);
        cuda_er = cudaDeviceSynchronize();
        assert(cuda_er == 0);

        if ((iter % 150 == 0) || (iter == 1))
        {
            difference(arr_dif, arr_d, arr2_d, start_arr, end_arr, stream, out_max_data);
            cuda_er = cudaMemcpyAsync(arr_difh, arr_dif, (size + bias_1) * (loc_y + bias_halo) * sizeof(double), cudaMemcpyDeviceToHost,stream);
            assert(cuda_er == 0);
        }

        cuda_er = cudaEventSynchronize(compute_ok);
        mpi_er = MPI_Sendrecv(arr2_d + end_arr - ((size+bias_1)*2)+1, (size+bias_1), MPI_DOUBLE, next, 0,
                            arr2_d, (size+bias_1), MPI_DOUBLE, prev, 0, MPI_COMM_WORLD,
                            MPI_STATUS_IGNORE);
    
        mpi_er = MPI_Sendrecv(arr2_d + (size+bias_1), (size+bias_1), MPI_DOUBLE, prev, 0,
                                arr2_d + end_arr - (size+bias_1)+1, (size+bias_1), MPI_DOUBLE, next, 0, MPI_COMM_WORLD,
                                MPI_STATUS_IGNORE);
        assert(cuda_er == 0);

        if ((iter % 150 == 0) || (iter == 1))
        {
            cuda_er = cudaStreamSynchronize(stream);
            MPI_Barrier(MPI_COMM_WORLD);
            //mpi_er = MPI_Gather(arr_difh, part, MPI_DOUBLE, arr_table, full_part,MPI_DOUBLE, 0 , MPI_COMM_WORLD);
            cub_cmp(arr_dif, arr_d, arr2_d, start_arr, end_arr, stream, out_max_data);
            cudaMemcpy(&error, out_max_data, sizeof(double), cudaMemcpyDeviceToHost);
            if (rank == 0) printf("Iteration - %d ; Error = %0.14lf\n", iter, error);
        
        }

        
        ptr = arr_d;
        arr_d = arr2_d;
        arr2_d = ptr;
        
    }


    cuda_er = cudaMemcpy(arr_debug, arr2_d, (size + bias_1) * (loc_y + bias_halo) * sizeof(double), cudaMemcpyDeviceToHost);
    assert(cuda_er == 0);
    printf("------COPY MEMORY VRAM -> RAM (arr2_d -> arr_debug) SUCCESS!!!\n");

    // if ((rank == 0) || (rank == 1))
    // {
        int k=0;
        for (int i=0;i<(size + bias_1) * (loc_y + bias_halo);i++)
        {
            if (k==(size + bias_1))
            {
                printf("\n");
                k = 0;
            }
            k = k +1;
            //if (i == start_arr+ (size+bias_1)+1) arr_debug[i] =1;
            //if (i == start_arr+ (size+bias_1)+1+(size+bias_1)-3) arr_debug[i] =2;
            //if (i == end_arr - (size+bias_1)+ 2) arr_debug[i] =1;
            //if (i == end_arr - 1) arr_debug[i] =2;
            printf("%d ", (int)arr_difh[i]);
        }
        printf("rank = %d \n", rank);
    // }

    // cuda_er = cudaFree(&main_dev);
    // assert(cuda_er == 0);
    // printf("------FREE MEMORY (arr_d) SUCCESS!!!\n");

    // cuda_er = cudaFree(&arr_d);
    // assert(cuda_er == 0);
    // printf("------FREE MEMORY (arr_d) SUCCESS!!!\n");

    // cuda_er = cudaFree(&arr2_d);
    // assert(cuda_er == 0);
    // printf("------FREE MEMORY (arr_d) SUCCESS!!!\n");
    
    cudaEventDestroy(compute_ok);
    cudaStreamDestroy(stream);

    MPI_Finalize();
    printf("PROGRAM HAS ENDED SUCCESSFULLY!\n");
    return 0;
}