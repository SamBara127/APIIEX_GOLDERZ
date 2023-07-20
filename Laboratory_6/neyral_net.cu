#include "neyral_net.cuh"

__global__ void init_weight(float * weight, int size, int threads, int blocks)
{
    int i = (blockIdx.x*(threads*threads))  + (threads * threadIdx.y) + threadIdx.x;
    if (i<size)
    {
        weight[i] = 0.001;
    }
}

__global__ void bias(float* arr, float bias, int threads, int blocks,int size)
{
    int i = (blockIdx.x*(threads*threads))  + (threads * threadIdx.y) + threadIdx.x;
    if (i<size)
    {
        arr[i] = arr[i] + bias;
    }
}

__global__ void differ(float* arr1,float* arr2, int amount, int threads, int blocks, float * dif_arr)
{
    int i = (blockIdx.x*(threads*threads))  + (threads * threadIdx.y) + threadIdx.x;
    if (i<amount)
    {
        dif_arr[i] = arr1[i] - arr2[i];
    }
}

// scalar<<<BLOCKS2,THREADS2>>>(lay->gradient, lay->weights, diff_weights, size, lay->neyron_amount, threads, blocks);
__global__ void scalar(float* gradient,float* weights,float* diff_w, int size, int amount, int threads, int blocks)
{
    int i = blockIdx.y * threads + threadIdx.y;
    int j = blockIdx.x * threads + threadIdx.x;
    if (i<amount)
    {
        if (j<size)
        {
            diff_w[j + size*i] = gradient[i] * weights[j + size*i];
        }
    }
}


__global__ void scalarback(float* d_err,float* weights,float* in_data, int size, int amount, int threads, int blocks, float lr)
{
    int i = blockIdx.y * threads + threadIdx.y;
    int j = blockIdx.x * threads + threadIdx.x;
    if (i<amount)
    {
        if (j<size)
        {
            weights[j + size*i] = weights[j + size*i] + (in_data[j] * d_err[i]) * lr;
        }
    }
}


void Neyral::CuBLAS_Stat(cublasStatus_t cubl_stat)
{
    if (cubl_stat == CUBLAS_STATUS_ALLOC_FAILED) std::cout << "CUBLAS_STATUS_ALLOC_FAILED\n";
    if (cubl_stat == CUBLAS_STATUS_INVALID_VALUE) std::cout << "CUBLAS_STATUS_INVALID_VALUE\n";
    if (cubl_stat == CUBLAS_STATUS_ARCH_MISMATCH) std::cout << "CUBLAS_STATUS_ARCH_MISMATCH\n";
    if (cubl_stat == CUBLAS_STATUS_MAPPING_ERROR) std::cout << "CUBLAS_STATUS_MAPPING_ERROR\n";
    if (cubl_stat == CUBLAS_STATUS_EXECUTION_FAILED) std::cout << "CUBLAS_STATUS_EXECUTION_FAILED\n";
    if (cubl_stat == CUBLAS_STATUS_INTERNAL_ERROR) std::cout << "CUBLAS_STATUS_INTERNAL_ERROR\n";
}

void Neyral::CuDNN_Stat(cudnnStatus_t cubl_stat)
{
    if (cubl_stat == CUDNN_STATUS_ALLOC_FAILED) std::cout << "CUDNN_STATUS_ALLOC_FAILED\n";
    if (cubl_stat == CUDNN_STATUS_BAD_PARAM) std::cout << "CUDNN_STATUS_BAD_PARAM\n";
    if (cubl_stat == CUDNN_STATUS_ARCH_MISMATCH) std::cout << "CUDNN_STATUS_ARCH_MISMATCH\n";
    if (cubl_stat == CUDNN_STATUS_MAPPING_ERROR) std::cout << "CUDNN_STATUS_MAPPING_ERROR\n";
    if (cubl_stat == CUDNN_STATUS_EXECUTION_FAILED) std::cout << "CUDNN_STATUS_EXECUTION_FAILED\n";
    if (cubl_stat == CUDNN_STATUS_INTERNAL_ERROR) std::cout << "CUDNN_STATUS_INTERNAL_ERROR\n";
}

Neyral::Neyral()
{
    CuBLAS_Stat(cublasCreate_v2(&cublas));
    CuDNN_Stat(cudnnCreate(&cudnn));
    cudnn_er = cudnnCreateActivationDescriptor(&activDesc);
    assert(cudnn_er == 0);
    cudnn_er = cudnnSetActivationDescriptor(activDesc, CUDNN_ACTIVATION_SIGMOID,CUDNN_PROPAGATE_NAN,0.0);
    assert(cudnn_er == 0);
    cudnn_er = cudnnCreateTensorDescriptor(&in_desc);
    assert(cudnn_er == 0);
    cudnn_er = cudnnCreateTensorDescriptor(&out_desc);
    assert(cudnn_er == 0);
    cudnn_er = cudnnCreateTensorDescriptor(&dif_in_desc);
    assert(cudnn_er == 0);
    cudnn_er = cudnnCreateTensorDescriptor(&dif_out_desc);
    assert(cudnn_er == 0);
}

Layer* Neyral::AddLayer(int size_n, int cnt_weight)
{
    Layer* temp = new Layer;
    temp->neyron_amount = size_n;
    temp->size_layer = cnt_weight;

    cuda_er = cudaMalloc(&temp->in_data, cnt_weight * sizeof(float));
    assert(cuda_er == 0);

    cuda_er = cudaMalloc(&temp->weights, cnt_weight * size_n * sizeof(float));
    assert(cuda_er == 0);
    cuda_er = cudaMalloc(&temp->diff_weights, cnt_weight * size_n * sizeof(float));
    assert(cuda_er == 0);
    cuda_er = cudaMalloc(&temp->res, size_n * sizeof(float));
    assert(cuda_er == 0);
    cuda_er = cudaMalloc(&temp->res_active, size_n * sizeof(float));
    assert(cuda_er == 0);
    cuda_er = cudaMalloc(&temp->dif_res, size_n * sizeof(float));
    assert(cuda_er == 0);
    cuda_er = cudaMalloc(&temp->gradient, size_n * sizeof(float));
    assert(cuda_er == 0);

    cuda_er = cudaMemset(temp->diff_weights,0, cnt_weight * size_n * sizeof(float));
    assert(cuda_er == 0);

    float * rnd = (float*)malloc(cnt_weight * size_n * sizeof(float));
    int threads = 32;
    int blocks = cnt_weight * size_n;
    int mod_bl = blocks % (threads*threads);
    blocks = (mod_bl == 0) ? blocks/(threads*threads) : blocks/(threads*threads)+1;

    dim3 THREADS(threads,threads);
    dim3 BLOCKS(blocks);

    for(int i = 0;i<cnt_weight * size_n;i++)
    {
        rnd[i] = float(rand()) / (10*float(RAND_MAX))-(0.5/10);
    }
    cuda_er = cudaMemcpy(temp->weights, rnd, cnt_weight * size_n * sizeof(float), cudaMemcpyHostToDevice);
    assert(cuda_er == 0);

    
    //init_weight<<<BLOCKS,THREADS>>>(temp->weights,cnt_weight * size_n, threads, blocks);

    return temp;
}

float* Neyral::Forward(Layer* lay,float* data)
{

    cuda_er = cudaMemcpy(lay->in_data, data, lay->size_layer * sizeof(float), cudaMemcpyDeviceToDevice);
    assert(cuda_er == 0);



    CuBLAS_Stat(cublasSgemv(cublas, CUBLAS_OP_T, lay->size_layer,lay->neyron_amount,&alpha, lay->weights, lay->size_layer, data, 1, &beta, lay->res, 1));

    int threads = 32;
    int blocks = lay->size_layer;
    int mod_bl = blocks % (threads*threads);
    blocks = (mod_bl == 0) ? blocks/(threads*threads) : blocks/(threads*threads)+1;

    dim3 THREADS(threads,threads);
    dim3 BLOCKS(blocks);

    bias<<<BLOCKS,THREADS>>>(lay->res,(0.5),threads, blocks, lay->neyron_amount);

    cudnn_er = cudnnSetTensor4dDescriptor(in_desc,tensor_format,cudnn_type, 1, lay->neyron_amount, 1,1);
    assert(cudnn_er == 0);
    cudnn_er = cudnnSetTensor4dDescriptor(out_desc,tensor_format, cudnn_type, 1,lay->neyron_amount,1,1);
    assert(cudnn_er == 0);

    cudnn_er = cudnnActivationForward(cudnn, activDesc,&alpha,in_desc,lay->res,&beta,out_desc,lay->res_active);
    assert(cudnn_er == 0);
    return lay->res_active;
}

float* Neyral::Backward(Layer* lay,float* right, float loss, bool last)
{
    ///////////////////////////////first
    int threads = 32;
    int blocks = lay->neyron_amount;
    int mod_bl = blocks % (threads*threads);
    blocks = (mod_bl == 0) ? blocks/(threads*threads) : blocks/(threads*threads)+1;

    dim3 THREADS(threads,threads);
    dim3 BLOCKS(blocks);

    if (last == true) differ<<<BLOCKS,THREADS>>>(right, lay->res_active, lay->neyron_amount, threads, blocks,lay->dif_res);
    else 
    {
        cuda_er = cudaMemcpy(lay->dif_res, right, lay->neyron_amount * sizeof(float), cudaMemcpyDeviceToDevice);
        assert(cuda_er == 0);
    }
    ////////////////////////////////////////////////next
    int blocks_x = lay->size_layer;
    mod_bl = blocks_x % threads;
    blocks_x = (mod_bl == 0) ? blocks_x/threads : (blocks_x/threads)+1;

    int blocks_y = lay->neyron_amount;
    mod_bl = blocks_y % threads;
    blocks_y = (mod_bl == 0) ? blocks_y/threads : (blocks_y/threads)+1;

    dim3 THREADS2(threads,threads);
    dim3 BLOCKS2(blocks_x,blocks_y);

    cudnn_er = cudnnSetTensor4dDescriptor(in_desc,tensor_format,cudnn_type, 1, lay->neyron_amount, 1,1);
    assert(cudnn_er == 0);
    cudnn_er = cudnnSetTensor4dDescriptor(out_desc,tensor_format, cudnn_type, 1,lay->neyron_amount,1,1);
    assert(cudnn_er == 0);

    cudnn_er = cudnnSetTensor4dDescriptor(dif_in_desc,tensor_format,cudnn_type, 1, lay->neyron_amount, 1,1);
    assert(cudnn_er == 0);
    cudnn_er = cudnnSetTensor4dDescriptor(dif_out_desc,tensor_format, cudnn_type, 1,lay->neyron_amount,1,1);
    assert(cudnn_er == 0);

    cudnn_er = cudnnActivationBackward(cudnn, activDesc,&alpha,out_desc,lay->res_active,dif_in_desc,lay->dif_res,in_desc,lay->res,&beta,dif_out_desc,lay->gradient);
    assert(cudnn_er == 0);

    scalar<<<BLOCKS2,THREADS2>>>(lay->gradient, lay->weights, lay->diff_weights, lay->size_layer, lay->neyron_amount, threads, blocks);

    scalarback<<<BLOCKS2,THREADS2>>>(lay->gradient, lay->weights, lay->in_data, lay->size_layer, lay->neyron_amount, threads, blocks, loss);

    return lay->diff_weights;
}