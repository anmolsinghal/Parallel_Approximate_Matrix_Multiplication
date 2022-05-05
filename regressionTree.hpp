#ifndef REGRESSIONTREE_HPP
#define REGRESSIONTREE_HPP
#include "train.hpp"

// #include <unistd.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <unordered_map>
using namespace std;

#define NUM_LEVELS 4
#define NUM_LEAVES 16
#define NUM_NODES 15


class RegressionTree
{
private:
    // double thresholds[NUM_NODES];
    // int indices[NUM_LEVELS];
    // double prototypes[NUM_LEAVES][PROTOTYPE_DIM];
    double* thresholds;
    int* indices;
    double* prototypes;
    int D; // dimension of each example
    int C; // number of subspaces
    int M; // share of dimension of each subspace
    int R;
    double* products; 
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

    void fit(double** A_train, int N) {

        vector<int> root;
        for(int i = 0;i< N;i++)
            root.push_back(i);

        printf("Number of subspaces %d, share of each subspace %d\n", C, M);

        for(int c = 0; c < C; c++) {

            vector<vector<int>> cur_level;
            cur_level.push_back(root);
            
            for(int i = 0; i < NUM_LEVELS; i++) {
                printf("\nAt subspace %d, level %d, cur_level size is %lu\n", c, i, cur_level.size());
                vector<vector<int>> next_level;

                int j = heuristic_select_split_idx(A_train, cur_level, N, M, c * M);
                printf("found optimal split %d\n", j);

                indices[c*NUM_LEVELS + i] = j;

                for(int k = 0; k < cur_level.size(); k++)
                {   
                    auto b = cur_level[k];
                    double threshold =  optimal_split_unoptimised(A_train, b, j);
                    printf("found optimal threshold for node %d\n", (int)pow(2,i)-1+k);
                    thresholds[ c*NUM_NODES + (int)pow(2,i)-1+k ] = threshold; // TODO fix index

                    vector<int> left;
                    vector<int> right;

                    for(auto idx : b)
                    {
                        if(A_train[j][idx] <= threshold)
                            left.push_back(idx);
                        else
                            right.push_back(idx);
                    }

                    next_level.push_back(left);
                    next_level.push_back(right);
                }

                cur_level = next_level;

            } // built all levels for current subspace c
            printf("built all levels for current subspace, final cur level size is %lu\n", cur_level.size());

            // calculate prototypes for c
            // at this stage, cur_level.size() == NUM_LEAVES
            for(int i = 0; i < cur_level.size(); i++) {   
                for(int idx : cur_level[i]) { // while optimizing, prototypes could be col major ?   

                    // each prototype is built from the share of j indices that belong to current subspace c
                    for(int j = 0; j < M; j++) {
                        // prototypes[c][i][j] +=  A_train[j][idx];
                        prototypes[(c*NUM_LEAVES + i)*M + j] += A_train[j][idx];
                    }
                    
                }

                for(int j = 0; j < M; j++) {
                    // prototypes[c][i][j] /=  cur_level[i].size();
                    prototypes[(c*NUM_LEAVES + i)*M + j] /= cur_level[i].size();
                }
            }

        }// loop for all subspaces ends here


    }

    void print_prototypes() {
        int prototype_size = NUM_LEAVES * M;
        for(int c = 0; c < C; c++) {
            printf("Subspace #%d:\n", c);
            for(int i = 0; i < NUM_LEAVES; i++) {
                printf("\tPrototpye #%d: ", i);
                for(int j = 0; j < M; j++) {
                    // printf("%lf ", prototypes[i][j]);
                    printf("%lf ", prototypes[(c*NUM_LEAVES + i)*M + j]);
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
        for(int i =0;i<C;i++)
        {   
            for(int j =0;j<R;j++)
            {
                for(int k =0;k<NUM_LEAVES;k++)
                {
                    double product = dot_product( prototypes + (i*NUM_LEAVES+k)*M, B + j*D + i*M, M); 
                    //set value of product at ith subspace, kth leaf, and jth col
                    products[i*NUM_LEAVES*R + k*R + j ] = product;
                }
            }
        }
    }
    
    int get_leaf_index(double** in, int row, int c )
    {
        int cur_index = 0;
        
        for(int i =0;i< NUM_LEVELS;i++)
        {
            int b = in[row][c*NUM_LEVELS + indices[i]] >= thresholds[c*NUM_NODES + cur_index];

            cur_index = 2*cur_index + 1 + b;
        }

        return cur_index;
    }

    //Dimeniosn of input is n*D, output is n*R
    void predict(double** input, int n, double**& output)
    {
        for(int i =0;i<n;i++)
        {
            for(int j =0;j<R;j++)
            {
                for(int k= 0;k<C;k++)
                {
                    int leaf = get_leaf_index(input, i, k);
                    double product = products[k*NUM_LEAVES*R + leaf*R + j ];
                    output[i][j] = product;
                }
            }
        }
    }


    

};

#endif
