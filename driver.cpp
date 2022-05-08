#include "datagenerator.hpp"
#include "regressionTree.hpp"
#include "train.hpp"
#include "utils.hpp"
#include <omp.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// g++ -std=c++11 -fopenmp driver.cpp -o test && ./test

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

    printf("Train set size %d, number of subspaces %d\n", N, C);
    printf("Running %d omp threads\n", nthreads);

    Timer timer;

    RegressionTree* t = new RegressionTree(D, C);
    
    t->fit(A_train, N);
    printf("Built regression tree\n");
    // t->print_prototypes();

    t->precompute_products(B, R);
    // t->print_products();

    double serial_time = 0.0, omp_time = 0.0;

    // int N_test = 1000; // number of examples (rows) in test matrix A
    printf("Test matrix size\tOMP Time\tSerial Time\tSpeedup\t\tError\n");
    for(unsigned long N_test = N; N_test <= 15000; N_test += 1000) {
        printf("\n%lu x %d\t      ", N_test, D);
        double* A_test = generateExamples(N_test, D);
        
        omp_time = 0;
        double* output_cpu = new double[N_test * R];
        for(int i =0;i<N_test * R;i++)
        output_cpu[i] = 0;
        for(int i = 0; i < NUM_RUNS; i++) {
            timer.tic();
            t->predict_cpu(A_test, N_test, output_cpu);
            omp_time += timer.toc();
        }
        printf("%10lf\t", omp_time/NUM_RUNS);


        serial_time = 0;
        double* output = new double[N_test * R];
        for(int i =0;i<N_test * R;i++)
        output[i] = 0;
        for(int i = 0; i < NUM_RUNS; i++) {
            timer.tic();
            t->predict(A_test, N_test, output);
            serial_time += timer.toc();
        }
        printf("%10lf\t", serial_time/NUM_RUNS);

        printf("%10lf\t", serial_time / omp_time);
        
        // error
        double max_err = 0;
        for (long i = 0; i < N_test * R; i++) max_err = max(max_err, fabs(output[i] - output_cpu[i]));
        printf("%10e", max_err);

        delete [] output;
        delete [] output_cpu;
    }

    // print predicted output
    // printf("\nApproximate product:\n{");
    // print_matrix(output, N_test, R);

    // calculate actual output
    // double* output_ref = new double[N_test * R];

    // for(int i = 0; i < R; i++) {
    //     for(int j = 0; j < N_test; j++) {
    //         output_ref[i*N_test + j] = 0.0;
    //         for(int k = 0; k < D; k++) {
    //             output_ref[i*N_test + j] += A_test[k*N_test + j] * B[i*D + k];
    //         }
    //     }
    // }
}
