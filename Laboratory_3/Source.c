#include <stdlib.h>
#include <stdio.h> 
#include <malloc.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


// лямбда функция максимума для редукции
#define max(x, y) ((x) > (y) ? (x) : (y) )

int main(int argc, char *argv[])
{
	assert(argc == 4);
	//объявляем "size of matrix", "accurancy" и "iterations_max"
	clock_t start, end;
	double result;

	start = clock();
	int size = atoi(argv[1]);
	double accur = atof(argv[2]);
	int iter_max = atoi(argv[3]);
	
	//смещение массива на 2 индекса:
    // 1 - в связи с тем что происходит обрез тензора по "правому" и "нижнему" краю(<20 а не =20)
    // 2 - в свзи с тем что при пробеге в циклах по тензору по "правому" и "нижнему" краю будет браться
    // мнимый ноль на границах чтобы не выйти за предел тензора (см схему ниже под кодом)
	int bias_1 = 2;
	

	double* main_arr= (double*)malloc(((size + bias_1)*(size+bias_1)) * sizeof(double));
    double* main_arr2 = (double*)malloc(((size + bias_1)*(size+bias_1)) * sizeof(double));
	// объявление переменной отсчета итерации и градиента = шага нарастания от одной вершины до другой  
	int iter = 0;
	double gradient = 10.0 / size;
	// объявление переменной ошибки 
	double error = 1.0;
    // двойной указатель на массив как контейнер обмена данными между 2 тензорами
    double *ptr;
    //double buff;

	#pragma acc enter data create(main_arr[0:(size + bias_1)*(size+bias_1)],main_arr2[0:(size + bias_1)*(size+bias_1)]) copyin(size, gradient, bias_1)
    {
        #pragma acc parallel async(1) 
        {
            #pragma acc loop independent
            for (int i=0;i<size + bias_1;i++)
            {
                main_arr[i*(size+2) + 0] = 10 + gradient*i;
                main_arr[i] = 10 + gradient*i;
                main_arr[(size+1)*(size+2) + i] = 20 + gradient*i;
                main_arr[i*(size+2)+size+1] = 20 + gradient*i;
            }
        }
        #pragma acc parallel
        {
            #pragma acc loop independent
            for (int i=0;i<size + bias_1;i++)
            {
                main_arr2[i*(size+2) + 0] = main_arr[i*(size+2) + 0];
                main_arr2[i] = main_arr[i];
                main_arr2[(size+1)*(size+2) + i] = main_arr[(size+1)*(size+2) + i];
                main_arr2[i*(size+2)+size+1] = main_arr[i*(size+2)+size+1];
            }
        }
        #pragma acc wait(1)
    }

    cublasHandle_t handle;
    
    cublasCreate(&handle);
    int index;
    double max;
    double alpha = (-1.0);

    while ((error > accur) && (iter<iter_max))
    {
        iter++;


        #pragma acc data present(main_arr, main_arr2)
        #pragma acc parallel async(2)
        {
            #pragma acc loop gang vector()
            for (int j = 1; j < size + 1; j++)
            {
                #pragma acc loop gang vector()
                for (int i = 1; i < size + 1; i++)
                {
                    main_arr2[i*(size+2)+j] = 0.25 * (main_arr[(i+1)*(size+2)+j] + main_arr[(i-1)*(size+2)+j] + main_arr[i*(size+2)+j-1] + main_arr[i*(size+2)+j+1]);
                }
            }
        }

        if ((iter % 150 == 0) || (iter == 1))
        {
            #pragma acc wait(2)
            
            #pragma acc host_data use_device(main_arr, main_arr2)
            {
                cublasDaxpy(handle, ((size+2)*(size+2)), &alpha, main_arr2, 1, main_arr, 1);
                cublasIdamax(handle, ((size+2)*(size+2)), main_arr, 1, &index);
            }
            
            #pragma acc update host(main_arr[index-1:1])
            max = main_arr[index-1];
            if (max < 0)  max = max *(-1);

            #pragma acc host_data use_device(main_arr, main_arr2)
            {
                cublasDcopy(handle, ((size+2)*(size+2)), main_arr2, 1, main_arr, 1);
            }

            error = max;
            printf("Iteration - %d ; Error = %0.14lf\n", iter, error);
        }


        ptr = main_arr;
        main_arr = main_arr2;
        main_arr2 = ptr;

    }

    cublasDestroy(handle);

	end = clock();
    result  = (double)(end-start);
    result = result / 1000000;
    printf("Iteration - %d ; Error = %0.14lf\n", iter, error);
    printf("Time = %f Seconds\n", result);

	return 0;

}

/*  схема строения тензора: 8 x 8

g(i) = 10 + gradient*i
f(i) = 20 + gradient*i
i = index

i = {0 .. 7} - 8 элементов
i = i + 2 = {0 .. 9} элементов - 7-ой элемент не равен значению вершины тензора
поэтому мы делаем +1 смещение для получения полного тензора без обреза по краям снизу и справа (index = 8);
далее делаем еще одно смещение +1 где мы получаем два мнимых края тензора c рандомными значеним 
чтобы при применении уравнения теплопроводности на точках (x, 8) и (8, x) где x -> {0 .. 8}
мы не заходили за пределы тензора или за область допустимой памяти:

main_arr2[i][j] = 0.25 * (main_arr[i+1][j] + main_arr[i-1][j] + main_arr[i][j-1] + main_arr[i][j+1]);


_i_|_0 _|_1 _|_2 _|_3 _|_4 _|_5 _|_6 _|_7 _|_8 _|_9 _
_0_|_10_|_1g_|_2g_|_3g_|_4g_|_5g_|_6g_|_7g_|_20_|_??_
_1_|_1g_|_x _|_x _|_x _|_x _|_x _|_x _|_x _|_1f_|_??_
_2_|_2g_|_x _|_x _|_x _|_x _|_x _|_x _|_x _|_2f_|_??_
_3_|_3g_|_x _|_x _|_x _|_x _|_x _|_x _|_x _|_3f_|_??_
_4_|_4g_|_x _|_x _|_x _|_x _|_x _|_x _|_x _|_4f_|_??_
_5_|_5g_|_x _|_x _|_x _|_x _|_x _|_x _|_x _|_5f_|_??_
_6_|_6g_|_x _|_x _|_x _|_x _|_x _|_x _|_x _|_6f_|_??_
_7_|_7g_|_x _|_x _|_x _|_x _|_x _|_x _|_x _|_7f_|_??_
_8_|_20_|_1f_|_2f_|_3f_|_4f_|_5f_|_6f_|_7f_|_30_|_??_
_9_|_??_|_??_|_??_|_??_|_??_|_??_|_??_|_??_|_??_|_??_

*/
