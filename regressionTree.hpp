#ifndef REGRESSIONTREE_HPP
#define REGRESSIONTREE_HPP
#include "train.hpp"

#include <omp.h>
#include <stdio.h>
#include <vector>
#include <cmath>
using namespace std;

#define NUM_LEVELS 4
#define NUM_LEAVES 16
#define NUM_NODES 15


class RegressionTree
{
private:
    double* thresholds;
    int* indices;
    double* prototypes;
    double* products; 
    int D; // dimension of each example
    int C; // number of subspaces
    int M; // share of dimension of each subspace
    int R; // number of columns in B

public:
    RegressionTree(int D, int C) 
    {
        this->D = D;
        this->C = C;
        M = D/C;
        thresholds = new double[C * NUM_NODES]; // C x NUM_NODES
        indices = new int[C * NUM_LEVELS]; // C x NUM_LEVELS
        prototypes = new double[C * NUM_LEAVES * M]; // C x NUM_LEAVES x C
    }

    void fit(double* A_train, int N) {

        vector<int> root;
        for(int i = 0;i< N;i++)
            root.push_back(i);

        printf("Number of subspaces %d, share of each subspace %d\n", C, M);

        for(int c = 0; c < C; c++) {

            vector<vector<int>> cur_level;
            cur_level.push_back(root);
            
            for(int i = 0; i < NUM_LEVELS; i++) {
                // printf("\nAt subspace %d, level %d, cur_level size is %lu\n", c, i, cur_level.size());
                vector<vector<int>> next_level;

                int j = heuristic_select_split_idx(A_train, cur_level, N, M, c * M);
                // printf("found optimal split %d\n", j);

                indices[c*NUM_LEVELS + i] = j;

                for(int k = 0; k < cur_level.size(); k++)
                {   
                    auto b = cur_level[k];
                    double threshold =  optimal_split_unoptimised(A_train, N, b, j);
                    // printf("found optimal threshold for node %d\n", (int)pow(2,i)-1+k);
                    thresholds[ c*NUM_NODES + (int)pow(2,i)-1+k ] = threshold; // TODO fix index

                    vector<int> left;
                    vector<int> right;

                    for(auto idx : b)
                    {
                        if(A_train[j*N + idx] <= threshold)
                            left.push_back(idx);
                        else
                            right.push_back(idx);
                    }

                    next_level.push_back(left);
                    next_level.push_back(right);
                }

                cur_level = next_level;

            } // built all levels for current subspace c
            // printf("built all levels for subspace %d, final cur level size is %lu\n", c, cur_level.size());

            // calculate prototypes for c
            // at this stage, cur_level.size() == NUM_LEAVES
            for(int i = 0; i < cur_level.size(); i++) {   
                for(int idx : cur_level[i]) { // while optimizing, prototypes could be col major ?   

                    // each prototype is built from the share of j indices that belong to current subspace c
                    for(int j = 0; j < M; j++) {
                        // prototypes[c][i][j] +=  A_train[j][idx];
                        // prototypes[(c*NUM_LEAVES + i)*M + j] += A_train[j*N + idx];
                        prototypes[c*NUM_LEAVES*M + j*NUM_LEAVES + i] += A_train[j*N + idx];
                    }
                    
                }

                for(int j = 0; j < M; j++) {
                    // prototypes[c][i][j] /=  cur_level[i].size();
                    // prototypes[(c*NUM_LEAVES + i)*M + j] /= cur_level[i].size();
                    prototypes[c*NUM_LEAVES*M + j*NUM_LEAVES + i] /= cur_level[i].size();
                }
            }

        }// loop for all subspaces ends here


    }

    void print_prototypes() {
        int prototype_size = NUM_LEAVES * M;
        for(int c = 0; c < C; c++) {
            printf("Subspace #%d:\n", c);
            // printf("{");
            for(int i = 0; i < NUM_LEAVES; i++) {
                printf("\tPrototpye #%d: ", i);
                // printf("{");
                for(int j = 0; j < M; j++) {
                    // printf("%lf ", prototypes[i][j]);
                    printf("%lf ", prototypes[(c*NUM_LEAVES + i)*M + j]);
                }
                printf("\n");
                // printf("}");
            }
            printf("\n");
            // printf("}");
        }
        printf("\n");
    }

    void print_products() {
        int product_size = NUM_LEAVES * R;
        for(int c = 0; c < C; c++) {
            printf("Subspace #%d:\n", c);
            for(int i = 0; i < NUM_LEAVES; i++) {
                printf("\tPrecomputed Product #%d: ", i);
                for(int j = 0; j < R; j++) {
                    // printf("%lf ", prototypes[i][j]);
                    printf("%lf ", products[(c*NUM_LEAVES + i)*R + j]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }
    
    //dimension of B is D*R
    void precompute_products(double* B, int R)
    {   
        this->R = R;
        products = new double[C * NUM_LEAVES * R];
        // initialize products array
        for(int i = 0; i < C * NUM_LEAVES * R; i++) products[i] = 0.0;

        for(int c = 0; c < C; c++)
        {   
            for(int j = 0; j < R; j++)
            {
                for(int k = 0; k < M; k++) {
                        
                    for(int i = 0; i < NUM_LEAVES; i++)
                    {
                        // double product = dot_product(prototypes + (c*NUM_LEAVES+i)*M, B + j*D + c*M, M);
                        // //set value of product at ith subspace, kth leaf, and jth col
                        // products[c*NUM_LEAVES*R + i*R + j ] += product;
                        products[c*NUM_LEAVES*R + j*NUM_LEAVES + i] += prototypes[c*NUM_LEAVES*M + k*NUM_LEAVES + i] * B[c*M + j*D + k]; // col major products
                    }
                }
            }
        }
    }
    
    int get_leaf_idx(double* in, int row, int c, int N)
    {
        int cur_index = 0;
        
        for(int i = 0; i < NUM_LEVELS-1; i++) {
            int b = in[indices[i] * N + row] >= thresholds[c*NUM_NODES + cur_index];
            cur_index = 2*cur_index + 1 + b;
        }

        return cur_index;
    }

    //Dimension of input is N*D, output is N*R
    void predict(double* input, int N, double* output) {        
        for(int j = 0; j < R; j++) {
            for(int c = 0; c < C; c++) {
                // unsigned long prod_offset = (unsigned long)c*NUM_LEAVES*R + j*NUM_LEAVES;
                for(int i = 0; i < N; i++) {
                    int leaf = get_leaf_idx(input, i, c, N);
                    // double product = products[c*NUM_LEAVES*R + leaf*R + j]; // row major products
                    double product = products[c*NUM_LEAVES*R + j*NUM_LEAVES + leaf]; // col major products
                    output[j * N + i] += product;
                }
            }
        }
    }

    void predict_cpu(double* input, int N, double* output)
    {
        #pragma omp parallel
        {
            #pragma omp for collapse(3)
            for(int j = 0; j < R; j++) {
                for(int c = 0; c < C; c++) {
                    // unsigned long prod_offset = (unsigned long)c*NUM_LEAVES*R + j*NUM_LEAVES;
                    for(int i = 0; i < N; i++) {
                        int leaf = get_leaf_idx(input, i, c, N);
                        // double product = products[c*NUM_LEAVES*R + leaf*R + j]; // row major products
                        double product = products[c*NUM_LEAVES*R + j*NUM_LEAVES + leaf]; // col major products
                        output[j * N + i] += product;
                    }
                }
            }
        }
    }
};

#endif
