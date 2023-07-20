#include "dataset_data.cuh"

__global__ void init_data(float * data, int size, int threads, int blocks)
{
    int i = (blockIdx.x*(threads*threads))  + (threads * threadIdx.y) + threadIdx.x;
    if (i<size)
    {
        //if (data[i] > 0) data[i] = 1;
        data[i] = data[i] / 256;
    }
}

Batch Dataset_data::CreateBatch(float* data, int size,float *out)
{
    Batch* temp = new Batch;
    cuda_er = cudaMalloc(&temp->data,size * sizeof(float));
    assert(cuda_er == 0);
    cuda_er = cudaMemcpy(temp->data,data,size * sizeof(float), cudaMemcpyHostToDevice);
    assert(cuda_er == 0);

    cuda_er = cudaMalloc(&temp->out, sizeof(float));
    assert(cuda_er == 0);
    cuda_er = cudaMemcpy(temp->out,out, sizeof(float), cudaMemcpyHostToDevice);
    assert(cuda_er == 0);
    //temp->out = out;

    int threads = 32;
    int blocks = size;
    int mod_bl = blocks % (threads*threads);
    blocks = (mod_bl == 0) ? blocks/(threads*threads) : blocks/(threads*threads)+1;

    dim3 THREADS(threads,threads);
    dim3 BLOCKS(blocks);

    init_data<<<BLOCKS,THREADS>>>(temp->data,size, threads, blocks);

    return *temp;
}

float* Dataset_data::GetDatasetData(int number)
{
    //return dat_h[number].data;
    return 0;
}

float* Dataset_data::GetDatasetOut(int number)
{
    //return dat_h[number].out;
    return 0;
}