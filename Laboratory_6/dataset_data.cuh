#pragma once
#include <malloc.h>
#include <assert.h>
#include "data_struct.hh"

class Dataset_data
{
private:
    size_t b_size;
    size_t num_b;
    float* data_device;
    float* out_device;
    cudaError cuda_er;

public:
    Batch CreateBatch(float* data, int size,float *out);
    float* GetDatasetData(int number);
    float* GetDatasetOut(int number);
};
