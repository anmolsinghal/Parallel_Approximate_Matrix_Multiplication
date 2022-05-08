#include "datagenerator.hpp"
#include "regressionTree.hpp"
#include "train.hpp"
#include "utils.hpp"
#include "predict_gpu.cu"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>

// nvcc -std=c++11 -Xcompiler -fopenmp driver.cu -o test && ./test

__host__
void print_matrix(double* mat, int rnum, int cnum) {
    for(int i = 0; i < rnum; i++) {
        printf("{");
        for(int j = 0; j < cnum; j++) {
            printf("%lf,", mat[j*rnum + i]);
        }
        printf("},");
    }
    printf("}\n");
}
__host__
void convert_to_row_major(double* input, double* output, int rows, int cols)
{
    for(int i =0;i<rows;i++)
    {
        for(int j = 0;j<cols;j++)
        {
            output[i*cols+j] = input[j*rows+i];
        }
    }
}
__host__
int main(int argc, char** argv) {
    int N = 1000; // number of examples in A_train
    int D = 256; // size of each example
    int C = 8; // number of subspaces
    int R = 32; // size of each row in B
    int NUM_RUNS = 5; // number of inference experiments to run
    
    int nthreads = 1;

    // handle arguments with getopt
    char *spec = NULL;
    int index;
    int c;
    char sched_algo;

    opterr = 0;

    while ((c = getopt (argc, argv, "n:d:r:c:")) != -1)
    switch (c)
    {
    case 'n':
        spec = optarg;
        sscanf(spec, "%d", &N);
        break;
    case 'd':
        spec = optarg;
        sscanf(spec, "%d", &D);
        break;
    case 'r':
        spec = optarg;
        sscanf(spec, "%d", &R);
        break;
    case 'c':
        spec = optarg;
        sscanf(spec, "%d", &C);
        break;

    case '?':
        if (optopt == 's')
            fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint (optopt))
            fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
            fprintf (stderr,
                    "Unknown option character `\\x%x'.\n",
                    optopt);
        exit(EXIT_FAILURE);
    
    default:
        printf("ERROR opt %c\n", c);
        exit(EXIT_FAILURE);
    }

    #pragma omp parallel
    {
        #pragma omp sections nowait
        {
            #pragma omp section
            {
                nthreads = omp_get_num_threads();
            }
        }
    }

    double* A_train = generateExamples(N, D);
    double* B = generateExamples(D, R);
    
    double* device_matrix, *device_products,  *device_thresholds, *device_output;
    int *device_indices;
	cudaMalloc( (void**)&device_products, C*NUM_LEAVES*R* sizeof(double));
	cudaMalloc( (void**)&device_indices, C*NUM_LEVELS* sizeof(int));
    cudaMalloc( (void**)&device_thresholds, C*NUM_NODES* sizeof(double));

    printf("Train set size %d, number of subspaces %d\n", N, C);
    printf("Running %d omp threads\n", nthreads);

    Timer timer;

    RegressionTree* t = new RegressionTree(D, C);
    
    t->fit(A_train, N);
    printf("Built regression tree\n");
    // t->print_prototypes();

    t->precompute_products(B, R);
    // t->print_products();

	cudaMemcpy((void*)device_products, (void*)t->products,C*NUM_LEAVES*R *sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((void*)device_indices, (void*)t->indices, C*NUM_LEVELS* sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy((void*)device_thresholds, (void*)t->thresholds,C*NUM_NODES* sizeof(double),cudaMemcpyHostToDevice);

    double serial_time, omp_time, gpu_time = 0.0;

    // int N_test = 1000; // number of examples (rows) in test matrix A
    printf("Test matrix size\tSerial Time\tOMP Time\tGPU Time\tOMP Speedup\tGPU Speedup\tOMP Error\tGPU Error\n");
    // printf("Test matrix size\tOMP Time\tSerial Time\tSpeedup\t\tError\n");
    for(unsigned long N_test = N; N_test <= 15000; N_test += 1000) {
        double* A_test = generateExamples(N_test, D);
        
        // calculate serial output
        serial_time = 0.0;
        double* output = new double[N_test * R];
        for(int i =0;i<N_test * R;i++)
        output[i] = 0;
        for(int i = 0; i < NUM_RUNS; i++) {
            timer.tic();
            t->predict(A_test, N_test, output);
            serial_time += timer.toc();
        }

        // calculate omp parallelized output
        omp_time = 0.0;
        double* output_cpu = new double[N_test * R];
        for(int i =0;i<N_test * R;i++)
        output_cpu[i] = 0;
        for(int i = 0; i < NUM_RUNS; i++) {
            timer.tic();
            t->predict_cpu(A_test, N_test, output_cpu);
            omp_time += timer.toc();
        }
        
        // absolute error omp vs serial
        double omp_max_err = 0;
        for (long i = 0; i < N_test * R; i++)
            omp_max_err = max(omp_max_err, fabs(output[i] - output_cpu[i]));

        // calculate gpu parallelized output
        double* A_test_row_major = new double[ N_test*D];
        // // double* A_test_double_transpose = new double[N_test*D];

        convert_to_row_major(A_test, A_test_row_major, N_test, D);
        cudaMalloc((void**)&device_matrix, N_test*D*sizeof(double));
        cudaMemcpy((void*)device_matrix, (void*)A_test_row_major, N_test*D* sizeof(double) ,cudaMemcpyHostToDevice);
        // // convert_to_row_major(A_test_row_major, A_test_double_transpose, D, N_test);
        // // max_err = 0;
        // // for (long i = 0; i < N_test * D; i++) max_err = max(max_err, fabs(A_test[i] - A_test_double_transpose[i]));
        // // printf("Error from transpose: %10e\n", max_err);
        
        cudaMalloc( (void**)&device_output, N_test*R*sizeof(double));
        // cudaMemset(device_output, 0, N_test*R*sizeof(double));
        cudaEvent_t start,stop;
        float elapsedTime;
        cudaEventCreate (&start);
        cudaEventCreate (&stop);

        dim3 dimGrid(N_test);
        dim3 dimBlock(R);
        cudaEventRecord (start, 0);
        gpu_time = 0.0;
        for(int i = 0; i < NUM_RUNS; i++) {
            cudaMemset(device_output, 0, N_test*R*sizeof(double));
        //     // double starttime = omp_get_wtime();
            timer.tic();
            for(int c = 0; c < C; c++) {
                predict_gpu<<<dimGrid, dimBlock>>>(device_matrix, N_test, D, R, D/C, device_products, device_indices, device_thresholds, device_output, c);
                cudaDeviceSynchronize();
            }
        //     // gpu_time += omp_get_wtime() - starttime;
            gpu_time += timer.toc();
        }

        double* host_result = new double[N_test*R];
        cudaMemcpy((void*)host_result, (void*)device_output,N_test*R*sizeof(double),cudaMemcpyDeviceToHost);

        cudaEventRecord (stop, 0);
        cudaEventSynchronize (stop);
        cudaEventElapsedTime ( &elapsedTime, start, stop);

        double gpu_max_error = 0;
        for (long i = 0; i < N_test * R; i++)
            gpu_max_error = max(gpu_max_error, fabs(host_result[i] - output[i]));

        delete [] output;
        delete [] output_cpu;
        delete [] A_test_row_major;
        delete [] host_result;
        cudaFree(device_output);

        //print all
        printf("\n%lu x %d\t      ", N_test, D);
        printf("%10lf\t", serial_time/NUM_RUNS);
        printf("%10lf\t", omp_time/NUM_RUNS);
        printf("%10lf\t", gpu_time/NUM_RUNS);
        printf("%10lf\t", serial_time / omp_time);
        printf("%10lf\t", serial_time / gpu_time);
        printf("%10e\t", omp_max_err);
        printf("%10e", gpu_max_error);
    }
    
}
