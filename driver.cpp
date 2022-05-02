#include "datagenerator.hpp"
#include "regressionTree.hpp"
#include "train.hpp"

int main(int argc, char** argv) {
    int N = 1000, D = 5;
    double** A_train = generateExamples(N, D);
    RegressionTree* t = new RegressionTree();
    t->fit(A_train, N, D);

    t->print_prototypes();
}
