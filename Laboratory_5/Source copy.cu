#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#define MAX_THREADS 32

__global__ void deb(double *arr, int threads, int blocks)
{
    long long int i = (blockIdx.x*(threads*threads)) + (threadIdx.x + threads * threadIdx.y);
    arr[i] = 90;
}

__global__ void init(double *arr, int size_x, int size_y, int threads, int blocks, double gradient, int rank, int size_rank)
{
    long long int i = (blockIdx.x*(threads*threads)) + (threadIdx.x + threads * threadIdx.y);
    if (rank == 0)
    {
        if ((i>=0) && (i<size_x-1)) arr[i] =  10 + gradient*i;
        if ((i % (size_x)) == 0) arr[i] =  10 + (gradient*i)/(size_x);
        if (((i+2) % (size_x)) == 0) arr[i] = 20 + (gradient*i)/(size_x);
    }
    else 
    {
        if ((i % (size_x)) == 0) arr[i] =  10 +(rank*()) + (gradient*i)/(size_x);
        if (((i+2) % (size_x)) == 0) arr[i] = 20 + (gradient*i)/(size_x);
    }
}

cudaError_t init_bound(double * arr_d, double* arr2_d, int rank, int size_rank, double gradient, int start, int end, int size_x, int size_y)
{
    int size = (end - start) ;
    int threads = MAX_THREADS;
    int int_blocks = size;
    int mod_blocks = int_blocks % (threads*threads);
    int_blocks = (mod_blocks == 0) ? int_blocks / (threads*threads) : (int_blocks / (threads*threads))+1;
    int blocks = int_blocks;

    dim3 THREADS(threads,threads);
    dim3 BLOCKS(blocks);
    deb<<<BLOCKS,THREADS>>>(arr_d, threads, blocks);
    init<<<BLOCKS,THREADS>>>(arr_d, size_x, size_y, threads, blocks, gradient, rank, size_rank);
    return cudaGetLastError();
}