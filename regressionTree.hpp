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
using namespace std;

#define d 128


struct regressionTree
{
    double thresholds[15];
    int indices[4];
    double prototypes[16][d];

    regressionTree(double ts[], int i[] )
    {
        
    }


    int find_index_serial(double input[])
    {
        int cur_index = 0;

        for(int i =0;i< 4;i++)
        {
            int b = (input[indices[i]] < thresholds[cur_index]) ? 1 : 0;
            cur_index = 2*cur_index - 1 + b;
        }

        return cur_index;
    }

};