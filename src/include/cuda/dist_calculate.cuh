#pragma once
#include <vector>
#include <cuda_runtime.h>

__global__ void euclideanKernel(char **data_, const float *baseline, int dim,
                                int numVecs, float *results);

__global__ void euclideanSingleKernel(float *vec, const float *baseline,
                                      int dim, float *result);

class DistanceCalculator {
  public:
    DistanceCalculator(const float *h_baseline, int dim,
                       cudaStream_t stream = 0);

    ~DistanceCalculator();

    std::vector<float> computeDistances(std::vector<char *> &data,
                                        cudaStream_t stream);

    float computeSingleDistanceAsync(char *d_vec, cudaStream_t stream = 0);

  private:
    float *d_baseline;
    int dimension;
};
