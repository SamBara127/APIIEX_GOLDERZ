#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h> 
#include <malloc.h>
#include <cuda.h>
#define MAX 512*512
#define MAX_THREADS 32

__global__ void kernel(long long int *arr, size_t size, int threads, int blocks)
{
    long long int i = (blockIdx.x*(threads*threads)) + (blockIdx.y * (threads*threads*blocks)) + (threadIdx.x + threads * threadIdx.y);
    arr[i] = i/2;
} 

int main()
{
    long long int arr[MAX];
    long long int *arr_d = NULL;
    clock_t start, end;
    double result = 0.0;
    double count_blocks = 0;
    int int_blocks = 0;
    int mod_blocks = 0;

    size_t size = MAX * sizeof(long long int);
    cudaError_t cuda_err;

    
    cuda_err = cudaMalloc(&arr_d, size);
    assert(cuda_err == 0);
    printf("ALLOCATED MEMORY SUCCESS!!!\n");

    int threads = MAX_THREADS;
    count_blocks = sqrt(MAX);
    int_blocks = (count_blocks == (int)count_blocks) ? (int)count_blocks :(int)count_blocks+1;
    mod_blocks = int_blocks % threads;
    int_blocks = (mod_blocks == 0) ? int_blocks / threads : (int_blocks / threads)+1;
    int blocks = int_blocks;

    dim3 THREADS(threads,threads);
    dim3 BLOCKS(blocks,blocks);
    start = clock();

    kernel<<<BLOCKS, THREADS>>>(arr_d, size, threads, blocks);

    end = clock();
    cuda_err = cudaMemcpy(arr, arr_d, size, cudaMemcpyDeviceToHost);
    assert(cuda_err == 0);
    printf("COPIOUT VRAM -> RAM SOCCESSFULLY!!!\n");

    cuda_err = cudaFree(arr_d);
    assert(cuda_err == 0);
    printf("FREE MEMORY SUCCESS!!!\n");

    result  = (double)(end-start);
    result = result / 1000000;
    for (int i=0;i<MAX;i++)
    {
        if (i%100 == 0) printf("%lli\n", arr[i]);
    }
    printf("%lli\n", arr[MAX-1]);
    printf("%lli\n", arr[MAX]);
    printf("Time = %f Seconds\n", result);
    return 0;
}