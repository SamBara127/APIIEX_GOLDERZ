#pragma once
#include "layer.hh"
#include "data_struct.hh"
#include <assert.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <cstdlib>

class Neyral
{
private:
    cublasStatus_t cubl_stat;
    cublasHandle_t cublas;

    cudnnStatus_t cudnn_er;
    cudnnHandle_t cudnn;
    cudnnActivationDescriptor_t activDesc;

    cudnnTensorDescriptor_t in_desc, out_desc, dif_in_desc, dif_out_desc;
    
    cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW;
    cudnnDataType_t cudnn_type = CUDNN_DATA_FLOAT;

    cudaError_t cuda_er;

    float alpha = 1.0f;
    float beta = 0.0f;

    void CuBLAS_Stat(cublasStatus_t cubl_stat);
    void CuDNN_Stat(cudnnStatus_t cubl_stat);
public:
    Neyral();
    Layer* AddLayer(int size_n, int cnt_weight);
    float* Forward(Layer* lay,float* data);
    float* Backward(Layer* lay, float* right, float loss, bool last);
};