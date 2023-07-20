#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <cub/cub.cuh>
#define MAX_THREADS 32

__global__ void init(double *arr, int size, int threads, int blocks, double gradient)
{
    long long int i = (blockIdx.x*(threads*threads)) + (blockIdx.y * (threads*threads*blocks)) + (threadIdx.x + threads * threadIdx.y);
	if (i<(size+2)*(size+1)-1) 
	{
		if ((i>=0) && (i<=size)) arr[i] =  10 + gradient*i;
		if ((i % (size+2)) == 0) arr[i] =  10 + (gradient*i)/(size+2);
		if ((i>=(size+2)*(size)) && (i<=(size+2)*(size+1))) arr[i] =  20 + (gradient*(i - (size+2)*(size)));
		if (((i+2) % (size+2)) == 0) arr[i] =  20 + (gradient*i)/(size+2);
	}
} 

__global__ void copyinit(double *main_arr, double *arr_d,int size_loc, int threads, int blocks, int rank, int size_x, int size_rank)
{
    long long int i = (blockIdx.x*(threads*threads)) + (blockIdx.y * (threads*threads*blocks)) + (threadIdx.x + threads * threadIdx.y);
    int diap;
    if (rank == size_rank-1) diap = size_loc - size_x*size_rank; else diap = size_loc - size_x*2;
    if (size_rank ==3) diap = size_loc - size_x*2;
    if ((i>=size_x)&&(i<size_loc-size_x))
    {
        arr_d[i] = main_arr[i-size_x+(diap*rank)];
    }
} 

__global__ void compute(double *arr1, double *arr2, int threads, int blocks, int size_x, int start, int end, int rank, int size_rank)
{
    long long int i = (blockIdx.x*(threads*threads)) + (threadIdx.x + threads * threadIdx.y);
    if (rank == 0)
    {
        if ((i >= size_x*2) && (i < end - size_x +1) && !(i % size_x == 0) && !(i % size_x == size_x-1))
        {
            arr2[i] = 0.25 * (arr1[i-size_x] + arr1[i+size_x] + arr1[i-1]+arr1[i+1]);
            //arr2[i] = 1;
        }
    }
    else if (rank == size_rank - 1)
    {
        if ((i >= size_x) && (i < end - (size_x*2) +1) && !(i % size_x == 0) && !(i % size_x == size_x-1))
        {
            arr2[i] = 0.25 * (arr1[i-size_x] + arr1[i+size_x] + arr1[i-1]+arr1[i+1]);
            //arr2[i] = 1;
        }
    }
    else
    {
        if ((i >= size_x) && (i < end - size_x +1) && !(i % size_x == 0) && !(i % size_x == size_x-1))
        {
            arr2[i] = 0.25 * (arr1[i-size_x] + arr1[i+size_x] + arr1[i-1]+arr1[i+1]);
            //arr2[i] = 1;
        }
    }
} 

__global__ void different(double *arr_dif, double *arr_d, double *arr2_d, int threads, int blocks, int size, double* out)
{
    long long int i = (blockIdx.x*(threads*threads)) + (threadIdx.x + threads * threadIdx.y);
    if (i<=size) arr_dif[i] = arr2_d[i] - arr_d[i];
} 

__global__ void cub_comp(double *arr_dif, double *arr_d, double *arr2_d, int threads, int blocks, int size, double* out)
{
    long long int i = (blockIdx.x*(threads*threads)) + (threadIdx.x + threads * threadIdx.y);
    typedef cub::BlockReduce<double,32,cub::BLOCK_REDUCE_WARP_REDUCTIONS, 32> BlockReduce;
    
    __shared__ typename BlockReduce::TempStorage temp_storage;

    *out = BlockReduce(temp_storage).Reduce(arr_dif[i], cub::Max());
} 

cudaError_t init_bound(double * main_dev, double *arr_d, double gradient,  int size_x, int size_rank, int rank, int start, int end)
{
    int size_local = end - start + 1;
    int threads = MAX_THREADS;
    int int_blocks = size_x;
    int mod_blocks = int_blocks % threads;
    int_blocks = (mod_blocks == 0) ? int_blocks / threads : (int_blocks / threads)+1;
    int blocks = int_blocks;

	// тип dim3 нужен для многомерного представления типа данных, в данном случае 2D, у нас блоки и потоки - тензоры
    dim3 THREADS(threads,threads);
    dim3 BLOCKS(blocks,blocks);

    init<<<BLOCKS,THREADS>>>(main_dev, size_x-2,threads,blocks,gradient);

    copyinit<<<BLOCKS,THREADS>>>(main_dev, arr_d, size_local,threads,blocks, rank,size_x,size_rank);

    return cudaGetLastError();
}


void ComputeTensor(double* arr_d,double* arr2_d, int size_x, int start, int end, int rank, int size_rank, cudaStream_t stream)
{
    int size_local = end - start + 1;
    int threads = MAX_THREADS;
    int int_blocks = size_local;
    int mod_blocks = int_blocks % (threads*threads);
    int_blocks = (mod_blocks == 0) ? int_blocks / (threads*threads) : (int_blocks / (threads*threads))+1;
    int blocks = int_blocks;

	// тип dim3 нужен для многомерного представления типа данных, в данном случае 2D, у нас блоки и потоки - тензоры
    dim3 THREADS(threads,threads);
    dim3 BLOCKS(blocks);

    compute<<<BLOCKS, THREADS, 0, stream>>>(arr_d, arr2_d, threads, blocks, size_x, start, end, rank, size_rank);

}

void difference(double* arr_dif,double* arr_d,double* arr2_d,int start,int end,cudaStream_t stream, double* out)
{
    int size_local = end - start + 1;
    int threads = MAX_THREADS;
    int int_blocks = size_local;
    int mod_blocks = int_blocks % (threads*threads);
    int_blocks = (mod_blocks == 0) ? int_blocks / (threads*threads) : (int_blocks / (threads*threads))+1;
    int blocks = int_blocks;

	// тип dim3 нужен для многомерного представления типа данных, в данном случае 2D, у нас блоки и потоки - тензоры
    dim3 THREADS(threads,threads);
    dim3 BLOCKS(blocks);

    different<<<BLOCKS, THREADS, 0, stream>>>(arr_dif, arr_d, arr2_d, threads, blocks, size_local, out);
}

void cub_cmp(double* arr_dif,double* arr_d,double* arr2_d,int start,int end,cudaStream_t stream, double* out)
{
    int size_local = end - start + 1;
    int threads = MAX_THREADS;
    int int_blocks = size_local;
    int mod_blocks = int_blocks % (threads*threads);
    int_blocks = (mod_blocks == 0) ? int_blocks / (threads*threads) : (int_blocks / (threads*threads))+1;
    int blocks = int_blocks;

	// тип dim3 нужен для многомерного представления типа данных, в данном случае 2D, у нас блоки и потоки - тензоры
    dim3 THREADS(threads,threads);
    dim3 BLOCKS(blocks);

    cub_comp<<<BLOCKS, THREADS, 0, stream>>>(arr_dif, arr_d, arr2_d, threads, blocks, size_local, out);
    
}
