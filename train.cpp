#include <vector>
#include <limits>
#include <list>
#include <algorithm>
using namespace std;


double sse_loss_at_index(double** A_train, vector<int>& B, int N, int D, int j)
{
    
    double idx_mean = 0.0;
    double idx_sum = 0.0;
    for(int i : B) {
        idx_sum += A_train[j][i];
    }
    idx_mean = idx_sum / B.size();

    double accumulate_sse = 0.0; // holds L(j, B) for current j
    double curr_sse;
    for(int i : B) {
        curr_sse = A_train[j][i] - idx_mean;
        accumulate_sse += curr_sse * curr_sse;
    }
    return accumulate_sse;
}

int sse_loss_per_bucket(double** A_train, vector<int>& B, int N, int D) {
/*
    Given a bucket B (a list of indices in A_train that belong to the bucket):
    Returns the best split index j that maximizes sse loss
    as described in the equation (7) in the paper
*/
    double min_sse = 0.0;
    int best_j;
    
    for(int j = 0; j < D; j++) {       

        double accumulate_sse = sse_loss_at_index(A_train, B, N, D, j);
       

        if(accumulate_sse > min_sse) best_j = j;
    }

    return best_j;
}

int sse_loss_across_buckets(double** A_train, vector<vector<int>>& B, int N, int D) {
/*
    Given a list of buckets B:
    Returns the best split index j that maximizes sse loss summed across all buckets
*/
    double min_sse = 0.0;
    int best_j;
    
    for(int j = 0; j < D; j++) {
        double accumulate_sse_across_buckets = 0.0;
        
        for(vector<int>& b : B) {
            double accumulate_sse = sse_loss_at_index(A_train, b, N, D, j);
            accumulate_sse_across_buckets += accumulate_sse;
        }

        if(accumulate_sse_across_buckets > min_sse) best_j = j;
    }

    return best_j;
}



double optimal_split_unoptimised(double** A_train, vector<int>& B, int N, int D, int j)
{   

    auto get_loss = [](list<double> left, list<double> right)
    {
        double lmean = 0, rmean = 0;

        for(double l : left)
        lmean+= l;

        for(double r :right)
        rmean += r;

        lmean /= left.size();
        rmean /= right.size();

        double loss = 0;

        for(double l : left)
        loss += (lmean - l)*(lmean - l);

        for(double r : right)
        loss += (rmean - r)*(rmean - r);

        return loss;
    };

    vector<double> column;

    for(int b :B)
    column.push_back(A_train[j][b]);

    sort(column.begin(), column.end());

    list<double> left;
    list<double> right;
    for(int i = 1;i<B.size();i++)
    right.push_back(column[i]);

    left.push_back(column[0]);

    int best_index = 0;

    double min_loss = get_loss(left, right);


    for(int i =1;i< B.size();i++)
    {
        double r = right.front();
        right.pop_front();

        left.push_back(r);

        double loss = get_loss(left, right);

        if(loss < min_loss)
        {
            min_loss = loss;
            best_index = i;
        }

    }

    if(best_index == column.size()-1)
    return column[best_index];

    return (column[best_index] + column[best_index+1])/2;

    
}



