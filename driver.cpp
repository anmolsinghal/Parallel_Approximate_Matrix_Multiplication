#include "datagenerator.hpp"
#include "regressionTree.hpp"
#include "train.hpp"

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
    int N = 100; // number of examples in A_train
    int D = 128; // size of each example
    int C = 8; // number of subspaces
    int R = 15; // size of each row in B
    double* A_train = generateExamples(N, D);
    double* B = generateExamples(D, R);

    // print b for testing
    // printf("{");
    // for(int i = 0; i < R; i++) {
    //     printf("{");
    //     for(int j = 0; j < D; j++) {
    //         printf("%lf,", B[i*D + j]);
    //     }
    //     printf("}");
    // }
    // printf("}\n");

    RegressionTree* t = new RegressionTree(D, C);
    
    t->fit(A_train, N);
    t->print_prototypes();

    t->precompute_products(B, R);
    t->print_products();

    int N_test = 2*N;
    double* A_test = generateExamples(N_test, D);
    double* output_cpu = new double[N_test * R];
    for(int i =0;i<N_test * R;i++)
    output_cpu[i] = 0;
    t->predict_cpu(A_test, N_test, output_cpu);


    double* output = new double[N_test * R];
    for(int i =0;i<N_test * R;i++)
    output[i] = 0;
    t->predict(A_test, N_test, output);

    // print predicted output
    // printf("\nApproximate product:\n{");
    // print_matrix(output, N_test, R);

    // calculate actual output
    double* output_ref = new double[N_test * R];


    for(int i = 0; i < R; i++) {
        for(int j = 0; j < N_test; j++) {
            output_ref[i*N_test + j] = 0.0;
            for(int k = 0; k < D; k++) {
                output_ref[i*N_test + j] += A_test[k*N_test + j] * B[i*D + k];
            }
        }
    }

    // error
    double max_err = 0;
    for (long i = 0; i < N_test * R; i++) max_err = max(max_err, fabs(output[i] - output_cpu[i]));
    printf(" %10e\n", max_err);
}
