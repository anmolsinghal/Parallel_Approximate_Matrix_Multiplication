#include <unistd.h>
#include <string>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <queue>
#include <deque>
#include <algorithm>
#include <vector>
#include "train.cpp"
#include <cmath>
#include <math.h>
using namespace std;

#define d 128
#define NUM_LEVELS 3
#define NUM_LEAVES 16
#define NUM_NODES 15


class regressionTree
{
    double thresholds[NUM_NODES];
    int indices[NUM_LEVELS];
    double prototypes[NUM_LEAVES][d];

    public:

    int find_index_serial(double input[])
    {
        int cur_index = 0;

        for(int i =0;i< NUM_LEVELS;i++)
        {
            int b = (input[indices[i]] < thresholds[cur_index]) ? 1 : 0;
            cur_index = 2*cur_index - 1 + b;
        }

        return cur_index;
    }


    void fit(double** A_train, int N, int D )
    {
        vector<int> root ;
        for(int i = 0;i< N;i++)
        root.push_back(i);

        vector<vector<int>> cur_level;

        cur_level.push_back(root);

        for(int i =0;i< NUM_LEVELS;i++)
        {
            vector<vector<int>> next_level;

            int j = sse_loss_across_buckets(A_train, cur_level, N, d);

            indices[i] = j;

            for(int k =0;k< cur_level.size();k++)
            {   
                auto b = cur_level[k];
                double threshold =  optimal_split_unoptimised(A_train, b, N, d, j);

                thresholds[ (int)pow(2,i)-1+k ] = threshold; //TO DO fix index

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

        }


        for(int i =0;i< cur_level.size();i++)
        {   

            for(int idx : cur_level[i])
            {   
                for(int j = 0;j<D;j++)
                {
                    prototypes[i][j] +=  A_train[j][idx];
                }
                
            }

            for(int j = 0;j<D;j++)
            {
                prototypes[i][j] /=  cur_level[i].size();
            }
        }
        //prototype is row major
    }

};