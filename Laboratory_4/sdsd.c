#include <math.h>
#include <stdio.h>
#define MAX 5000

int main()
{
    double count_blocks = 0;
    int int_blocks = 0;
    int mod_blocks = 0;

    int threads = 32;
    count_blocks = sqrt(MAX);
    printf("%lf\n",count_blocks);

    int_blocks = (count_blocks == (int)count_blocks) ? (int)count_blocks :(int)count_blocks+1;
    printf("%d\n",int_blocks);

    mod_blocks = int_blocks % threads;
    printf("%d\n",mod_blocks);

    int_blocks = (mod_blocks == 0) ? int_blocks / threads : (int_blocks / threads)+1;
    printf("%d\n",int_blocks);

    int blocks = int_blocks;
    printf("%d\n",blocks);
    return 0;
}