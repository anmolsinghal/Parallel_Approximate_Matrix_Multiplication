#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double** generateExamples(int N, int D) {
    double** A = new double*[D];
    srand(time(NULL));
    // stored in column major order
    for(int i = 0; i < D; i++) {
        A[i] = new double[N];
        for(int j = 0; j < N; j++) {
            A[i][j] = rand();
        }
    }

    return A;
}

int main() {
    int N = 10, D = 5;
    double** A_train = generateExamples(N, D);
    
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < D; j++) {
            printf("%lf ", A_train[j][i]);
        }
        printf("\n");
    }

    return EXIT_SUCCESS;
}
