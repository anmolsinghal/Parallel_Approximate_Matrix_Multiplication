#include<stdio.h>
#include<cuda.h>
#include <cub/cub.cuh>

__global__ void predict_gpu(double* deviceMatrix,int num_rows,int D, int num_cols, int M,  double* precomputed_products, int* indices, double* thresholds, double* output)
{
    int subspace = threadIdx.x;
    int row = blockIdx.x;
    int col = blockIdx.y;
    typedef cub::BlockReduce<int, 8> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;  

    int cur_index = 0;
    
    double thread_data;

    for(int i =0;i< NUM_LEVELS-1;i++)
    {
        int b = in[indices[subspace*NUM_LEVELS+i] + row*D] >= thresholds[subspace*NUM_NODES + cur_index];

        cur_index = 2*cur_index + 1 + b;
    }

    thread_data = precomputed_products[subspace*NUM_LEAVES*R + cur_index*R + col]]
    double answer = BlockReduce(temp_storage).Sum(thread_data);

    output[col * N + row] = answer;

}