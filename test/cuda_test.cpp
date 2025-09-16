#include <gtest/gtest.h>
#include "cuda/dist_calculate.cuh"

TEST(DistanceCalculatorTest, SingleDistance) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int dim = 4;
    int size_data_per_element = sizeof(float) * dim + 4; // 假设没有其他数据
    float query[] = {1.0, 2.0, 3.0, 4.0};
    DistanceCalculator calculator(query, dim, size_data_per_element);

    float data[] = {1.0, 2.0, 3.0, 5.0}; // 与query的距离应为1.0
    char *d_data;
    cudaMalloc(&d_data, size_data_per_element);
    cudaMemcpy(d_data, data, size_data_per_element, cudaMemcpyHostToDevice);
    std::vector<float> distance = calculator.computeDistances(d_data, 1, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_data);
    EXPECT_NEAR(distance[0], 1.0, 1e-5);
    cudaStreamDestroy(stream);
}

TEST(DistanceCalculatorTest, MultipleDistances) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int dim = 4;
    int size_data_per_element = sizeof(float) * dim; // 假设没有其他数据
    float query[] = {1.0, 2.0, 3.0, 4.0};
    DistanceCalculator calculator(query, dim, size_data_per_element);

    float data[] = {1.0, 2.0, 3.0, 5.0, 2.0, 3.0, 4.0, 5.0}; // 与query的距离应为1.0和2.0
    char *d_data;
    cudaMalloc(&d_data, size_data_per_element * 2);
    cudaMemcpy(d_data, data, size_data_per_element * 2, cudaMemcpyHostToDevice);
    std::vector<float> distance = calculator.computeDistances(d_data, 2, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_data);
    EXPECT_NEAR(distance[0], 1.0, 1e-5);
    EXPECT_NEAR(distance[1], 2.0, 1e-5);
    cudaStreamDestroy(stream);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}