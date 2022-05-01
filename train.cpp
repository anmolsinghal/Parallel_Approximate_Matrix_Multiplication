#include <vector>

using namespace std;

int sse_loss_per_bucket(double** A_train, vector<int>& B, int N, int D) {
/*
    Given a bucket B (a list of indices in A_train that belong to the bucket):
    Returns the best split index j that maximizes sse loss
    as described in the equation (7) in the paper
*/
    double min_sse = 0.0;
    int best_j;
    
    for(int j = 0; j < D; j++) {
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
            double idx_mean = 0.0;
            double idx_sum = 0.0;
            for(int i : b) {
                idx_sum += A_train[j][i];
            }
            idx_mean = idx_sum / b.size();

            double accumulate_sse = 0.0; // holds L(j, b) for current j and current b
            double curr_sse;
            for(int i : b) {
                curr_sse = A_train[j][i] - idx_mean;
                accumulate_sse += curr_sse * curr_sse;
            }

            accumulate_sse_across_buckets += accumulate_sse;
        }

        if(accumulate_sse_across_buckets > min_sse) best_j = j;
    }

    return best_j;
}
