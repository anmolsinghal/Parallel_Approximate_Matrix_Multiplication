#include "datagenerator.hpp"
#include "regressionTree.hpp"
#include "train.hpp"
#include "utils.hpp"
#include <omp.h>

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
    int R = 15; // size of each row in B
    int NUM_RUNS = 100; // number of inference experiments to run
    double* A_train = generateExamples(N, D);
    double* B = generateExamples(D, R);
    int nthreads = 1;

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

    printf("Running %d omp threads\n", nthreads);

    Timer timer;

    RegressionTree* t = new RegressionTree(D, C);
    
    t->fit(A_train, N);
    // t->print_prototypes();

    t->precompute_products(B, R);
    // t->print_products();

    double serial_time = 0.0, omp_time = 0.0;

    // int N_test = 1000; // number of examples (rows) in test matrix A
    printf("Test matrix size\tOMP Time\tSerial Time\tSpeedup\t\tError\n");
    for(unsigned long N_test = N; N_test <= 1000000000; N_test += 1000) {
        printf("\n%lu x %d\t      ", N_test, D);
        double* A_test = generateExamples(N_test, D);
        
        double* output_cpu = new double[N_test * R];
        for(int i =0;i<N_test * R;i++)
        output_cpu[i] = 0;
        for(int i = 0; i < NUM_RUNS; i++) {
            timer.tic();
            t->predict_cpu(A_test, N_test, output_cpu);
            omp_time += timer.toc();
        }
        printf("%10lf\t", omp_time/NUM_RUNS);


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
