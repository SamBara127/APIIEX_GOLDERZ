#include <time.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h> 
#include <malloc.h>
#define MAX 512*512

int main()
{
    long long int arr[MAX];
    clock_t start, end;
    double result = 0.0;

    start = clock();

    for (long long int i=0;i<MAX;i++)
    {
        arr[i] = i/2;
    }
    end = clock();
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