#include "dist_calculate.cuh"
#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>

int block_size;
int size_data_per_element;
int offsetData;

DistanceCalculator::DistanceCalculator(const float *h_baseline, int dim,
                                       int size_per_element,
                                       cudaStream_t stream)
    : dimension(dim), size_per_element(size_per_element) {
    cudaMalloc(&d_baseline, dim * sizeof(float));
    cudaMemcpyAsync(d_baseline, h_baseline, dim * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
}

DistanceCalculator::~DistanceCalculator() { cudaFree(d_baseline); }

std::vector<float>
DistanceCalculator::computeDistances(std::vector<char *> &data,
                                     cudaStream_t stream) {
    int numVecs = data.size();
    if (numVecs == 0) {
        return std::vector<float>();
    }
    std::vector<float> results(numVecs);
    // 使用thrust::device_vector来处理数据
    thrust::device_vector<char *> d_data(data.begin(), data.end());
    thrust::device_vector<float> d_results(numVecs);
    int threads = 256;
    euclideanKernel<<<numVecs, threads, 0, stream>>>(
        thrust::raw_pointer_cast(d_data.data()), d_baseline, dimension, numVecs,
        thrust::raw_pointer_cast(d_results.data()));
    cudaStreamSynchronize(stream);
    // 将结果从设备复制到主机
    thrust::copy(d_results.begin(), d_results.end(), results.begin());
    return results;
}

std::vector<float> DistanceCalculator::computeDistances(char *data, int numVecs,
                                    cudaStream_t stream) {
    std::vector<float> results(numVecs);
    // 使用thrust::device_vector来处理数据
    thrust::device_vector<float> d_results(numVecs);
    // 将数据复制到设备
    int threads = 256;
    euclideanKernel<<<numVecs, threads, 0, stream>>>(
        data, d_baseline, dimension, size_per_element, numVecs,
        thrust::raw_pointer_cast(d_results.data()));
    cudaStreamSynchronize(stream);
    // 将结果从设备复制到主机
    thrust::copy(d_results.begin(), d_results.end(), results.begin());
    return results;
}

float DistanceCalculator::computeSingleDistanceAsync(char *d_vec,
                                                     cudaStream_t stream) {
    int maxThreadsPerBlock = 1024; // 根据实际GPU调整
    int threads = min(maxThreadsPerBlock, dimension);
    int sharedMemSize = threads * sizeof(float);
    float *d_result;
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));

    euclideanSingleKernel<<<1, threads, sharedMemSize, stream>>>(
        (float *)d_vec, d_baseline, dimension, d_result);
    cudaStreamSynchronize(stream);
    float result;
    cudaMemcpyAsync(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost,
                    stream);
    cudaFree(d_result);
    return result;
}

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void euclideanKernel(char **data_, const float *baseline, int dim,
                                int numVecs, float *results) {
    int vecIdx = blockIdx.x; // 每个block处理一个向量
    int tid = threadIdx.x;
    float sum = 0.0f;

    float *vec = reinterpret_cast<float *>(const_cast<char *>(data_[vecIdx]));

    // 计算每个线程的局部平方差和
    for (int i = tid; i < dim; i += blockDim.x) {
        float diff = vec[i] - baseline[i];
        // 计算平方差
        sum += diff * diff;
    }

    // Warp级归约
    sum = warpReduceSum(sum);

    // 块内归约总和
    if (tid % 32 == 0) {
        __shared__ float sharedSum[32];
        int warpId = tid / 32;
        sharedSum[warpId] = sum;
        __syncthreads();

        if (warpId == 0) {
            float total = 0;
            for (int j = 0; j < (blockDim.x + 31) / 32; j++)
                total += sharedSum[j];
            results[vecIdx] = sqrtf(total);
        }
    }
}

// 单个向量专用内核，使用共享内存+完全展开式归约
__global__ void euclideanSingleKernel(float *vec, const float *baseline,
                                      int dim, float *result) {
    extern __shared__ float sharedSum[]; // 动态共享内存
    int tid = threadIdx.x;
    float sum = 0.0f;

    // 每个线程处理多个元素
    for (int i = tid; i < dim; i += blockDim.x) {
        float diff = vec[i] - baseline[i];
        sum += diff * diff;
    }
    sharedSum[tid] = sum;
    __syncthreads();

    // 归约求和（适用于任意blockDim.x的2次幂）
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedSum[tid] += sharedSum[tid + stride];
        }
        __syncthreads();
    }

    // 最终结果写回
    if (tid == 0) {
        *result = sqrtf(sharedSum[0]);
    }
}

__global__ void euclideanKernel(char *data_, const float *baseline, int dim, int size_per_element,
                                int numVecs, float *results) {
    int vecIdx = blockIdx.x; // 每个block处理一个向量
    int tid = threadIdx.x;
    float sum = 0.0f;

    float *vec = reinterpret_cast<float *>(data_ + vecIdx * size_per_element);

    // 计算每个线程的局部平方差和
    for (int i = tid; i < dim; i += blockDim.x) {
        float diff = vec[i] - baseline[i];
        // 计算平方差
        sum += diff * diff;
    }

    // Warp级归约
    sum = warpReduceSum(sum);

    // 块内归约总和
    if (tid % 32 == 0) {
        __shared__ float sharedSum[32];
        int warpId = tid / 32;
        sharedSum[warpId] = sum;
        __syncthreads();

        if (warpId == 0) {
            float total = 0;
            for (int j = 0; j < (blockDim.x + 31) / 32; j++)
                total += sharedSum[j];
            results[vecIdx] = sqrtf(total);
        }
    }
}