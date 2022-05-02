#include "datagenerator.hpp"
#include "regressionTree.hpp"
#include "train.hpp"

int main(int argc, char** argv) {
    int N = 1000; // number of examples in A_train
    int D = 128; // size of each example
    int C = 8; // number of subspaces
    double** A_train = generateExamples(N, D);
    RegressionTree* t = new RegressionTree(D, C);
    t->fit(A_train, N);

    t->print_prototypes();
}
