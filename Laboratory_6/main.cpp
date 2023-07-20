#include <cuda_runtime.h>
#include <malloc.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include "dataset_data.cuh"
#include "neyral_net.cuh"


int main(int argc, char* argv[]) 
{
    assert(argc == 5);
    assert(atof(argv[2]) >= 10);
	int size = atoi(argv[1]);
	int amount = atof(argv[2]);
    int max_epoch = atoi(argv[3]);
    float lr = atof(argv[4]);

    float* buff = (float*)malloc(size * sizeof(float));
    float* right = (float*)malloc(amount * sizeof(float));

    for (int j=0;j<amount;j++)
    {
        //if (j<amount/2) right[j] = 1; else right[j] = 0;
        if (j < 5) right[j] = 1; else right[j] = 0;
    }

    Dataset_data dataset;

    Batch * data;
    data = (Batch*)malloc(amount * sizeof(Batch));

    FILE** fstream;
    fstream = (FILE**)malloc(amount * sizeof(FILE*));

    fstream[0] = fopen("input_data/smile/1.txt", "r");
    fstream[1] = fopen("input_data/smile/2.txt", "r");
    fstream[2] = fopen("input_data/smile/3.txt", "r");
    fstream[3] = fopen("input_data/smile/4.txt", "r");
    fstream[4] = fopen("input_data/smile/5.txt", "r");
    
    fstream[5] = fopen("input_data/sad/1.txt", "r");
    fstream[6] = fopen("input_data/sad/2.txt", "r");
    fstream[7] = fopen("input_data/sad/3.txt", "r");
    fstream[8] = fopen("input_data/sad/4.txt", "r");
    fstream[9] = fopen("input_data/sad/5.txt", "r");

    fstream[10] = fopen("input_data/1.txt", "r");
    fstream[11] = fopen("input_data/2.txt", "r");
    fstream[12] = fopen("input_data/3.txt", "r");
    fstream[13] = fopen("input_data/4.txt", "r");


    for (int f=0;f<amount;f++)
    {
        for (int i = 0;i < size;i++) 
        {          
            fscanf(fstream[f], "%f", &buff[i]);
        }
        data[f] = dataset.CreateBatch(buff,size,&right[f]);
    }
    //float error;//,err2;
    float* dat_buf;

    Neyral neyral;

    Layer* lay_1 = neyral.AddLayer(1024, size);
    Layer* lay_2 = neyral.AddLayer(256, 1024);
    Layer* lay_3 = neyral.AddLayer(16, 256);
    Layer* lay_4 = neyral.AddLayer(1, 16);

    float* buf = (float*)malloc(1024*1025 * sizeof(float));
    cudaError_t cr;
    int epoch = 0;

    while (epoch < max_epoch)
    {
        for (int g=0;g<amount-4;g++)
        {
            dat_buf = neyral.Forward(lay_1, data[g].data);
            dat_buf = neyral.Forward(lay_2, dat_buf);
            dat_buf = neyral.Forward(lay_3, dat_buf);
            dat_buf = neyral.Forward(lay_4, dat_buf);

            dat_buf = neyral.Backward(lay_4, data[g].out, lr, true);
            dat_buf = neyral.Backward(lay_3, dat_buf, lr, false);
            dat_buf = neyral.Backward(lay_2, dat_buf, lr, false);
            dat_buf = neyral.Backward(lay_1, dat_buf, lr, false);
        }
        epoch++;
        printf("EPOCH -> %d\n",epoch);
    }
    
    for (int g=0;g<amount;g++)
    {
        dat_buf = neyral.Forward(lay_1, data[g].data);
        dat_buf = neyral.Forward(lay_2, dat_buf);
        dat_buf = neyral.Forward(lay_3, dat_buf);
        dat_buf = neyral.Forward(lay_4, dat_buf);

        cr = cudaMemcpy(buf, dat_buf, 1 * sizeof(float), cudaMemcpyDeviceToHost);
        assert(cr == 0);

        printf("Result[%d] = %f\n",g,buf[0]);
    }
        // dat_buf = neyral.Forward(lay_1, data[9].data);
        // dat_buf = neyral.Forward(lay_2, dat_buf);
        // dat_buf = neyral.Forward(lay_3, dat_buf);
        // dat_buf = neyral.Forward(lay_4, dat_buf);

        // cr = cudaMemcpy(buf, dat_buf, 1 * sizeof(float), cudaMemcpyDeviceToHost);
        // assert(cr == 0);

        // printf("Result[9] = %f\n",buf[0]);
        // cudaError_t cr = cudaMemcpy(buf, dat_buf, 1 * sizeof(float), cudaMemcpyDeviceToHost);
        // assert(cr == 0);

        // error = right[0] - buf[0];

        // for(int i=0;i<1;i++)
        // {
        //     printf("%f  ->  %f\n",buf[i], error);
        // }
    // }

    // dat_buf = neyral.Backward(lay_4, data[9].out, 1, true);
    // dat_buf = neyral.Backward(lay_3, dat_buf, 1, false);
    // dat_buf = neyral.Backward(lay_2, dat_buf, 1, false);
    // dat_buf = neyral.Backward(lay_1, dat_buf, 1, false);

    // cudaError_t cr = cudaMemcpy(buf, dat_buf, 1024* 1024  * sizeof(float), cudaMemcpyDeviceToHost);
    // assert(cr == 0);
    
    // for (int i =0;i< 16;i++) printf("%f\n",buf[i]);


    return 0;
}