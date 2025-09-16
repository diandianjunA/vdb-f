#include <gtest/gtest.h>
#include <thread>
#include "hnswlib/hnswlib.h"

TEST(HNSWTest, BasicOperations) {
    int dim = 4;
    int k = 2;
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> appr_alg(&space, 100);

    // 添加向量
    std::vector<float> vec1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> vec2 = {2.0, 3.0, 4.0, 5.0};
    std::vector<float> vec3 = {3.0, 4.0, 5.0, 6.0};
    appr_alg.addPoint((void*)vec1.data(), 1);
    appr_alg.addPoint((void*)vec2.data(), 2);
    appr_alg.addPoint((void*)vec3.data(), 3);

    // 查询向量
    std::vector<float> query = {1.5, 2.5, 3.5, 4.5};
    auto result = appr_alg.searchKnn(query.data(), k);

    // 检查结果
    std::vector<int> expected_ids = {2, 1}; // 最近的两个点应该是vec2和vec1
    for (size_t i = 0; i < expected_ids.size(); i++) {
        auto res = result.top();
        result.pop();
        EXPECT_EQ(res.second, expected_ids[i]);
    }
}

TEST(HNSWTest, MultiSearch) {
    int dim = 4;
    int k = 2;
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> appr_alg(&space, 100);
    // 添加向量
    std::vector<float> vec1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> vec2 = {2.0, 3.0, 4.0, 5.0};
    std::vector<float> vec3 = {3.0, 4.0, 5.0, 6.0};
    appr_alg.addPoint((void*)vec1.data(), 1);
    appr_alg.addPoint((void*)vec2.data(), 2);
    appr_alg.addPoint((void*)vec3.data(), 3);
    // 查询向量
    std::vector<std::vector<float>> queries = {
        {1.5, 2.5, 3.5, 4.5},
        {2.5, 3.5, 4.5, 5.5}
    };
    for (const auto& query : queries) {
        auto result = appr_alg.searchKnn(query.data(), k);
        // 检查结果
        std::vector<int> expected_ids;
        if (query[0] == 1.5) {
            expected_ids = {2, 1}; // 最近的两个点应该是vec2和vec1
        } else {
            expected_ids = {3, 2}; // 最近的两个点应该是vec3和vec2
        }
        for (size_t i = 0; i < expected_ids.size(); i++) {
            auto res = result.top();
            result.pop();
            EXPECT_EQ(res.second, expected_ids[i]);
        }
    }
}

TEST(HNSWTest, Filter) {
    int dim = 4;
    int k = 2;
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> appr_alg(&space, 100);
    // 添加向量
    std::vector<float> vec1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> vec2 = {2.0, 3.0, 4.0, 5.0};
    std::vector<float> vec3 = {3.0, 4.0, 5.0, 6.0};
    appr_alg.addPoint((void*)vec1.data(), 1);
    appr_alg.addPoint((void*)vec2.data(), 2);
    appr_alg.addPoint((void*)vec3.data(), 3);
    // 定义过滤器，只允许ID为2的点
    class MyFilter : public hnswlib::BaseFilterFunctor {
    public:
        bool operator()(hnswlib::labeltype id) const {
            return id == 2;
        }
    };
    MyFilter filter;
    // 查询向量
    std::vector<float> query = {1.5, 2.5, 3.5, 4.5};
    auto result = appr_alg.searchKnn(query.data(), k, &filter);

    // 检查结果
    std::vector<int> expected_ids = {2}; // 最近的两个点应该是vec2和vec1
    for (size_t i = 0; i < expected_ids.size(); i++) {
        auto res = result.top();
        result.pop();
        EXPECT_EQ(res.second, expected_ids[i]);
    }
}

TEST(HNSWTest, Update) {
    int dim = 4;
    int k = 2;
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> appr_alg(&space, 100, 16, 200, 100, true);

    // 添加向量
    std::vector<float> vec1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> vec2 = {2.0, 3.0, 4.0, 5.0};
    std::vector<float> vec3 = {3.0, 4.0, 5.0, 6.0};
    appr_alg.addPoint((void*)vec1.data(), 1);
    appr_alg.addPoint((void*)vec2.data(), 2);
    appr_alg.addPoint((void*)vec3.data(), 3);

    // 更新向量
    std::vector<float> vec2_updated = {10.0, 10.0, 10.0, 10.0};
    appr_alg.addPoint((void*)vec2_updated.data(), 2, true); // replace_deleted=true

    // 查询向量
    std::vector<float> query = {9.0, 9.0, 9.0, 9.0};
    auto result = appr_alg.searchKnn(query.data(), k);

    // 检查结果
    std::vector<int> expected_ids = {2, 3}; // 最近的两个点应该是更新后的vec2和vec3
    for (size_t i = 0; i < expected_ids.size(); i++) {
        auto res = result.top();
        result.pop();
        EXPECT_EQ(true, std::find(expected_ids.begin(), expected_ids.end(), res.second) != expected_ids.end());
    }
}

TEST(HNSWTest, Delete){
    int dim = 4;
    int k = 2;
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> appr_alg(&space, 100, 16, 200, 100, true);
    // 添加向量
    std::vector<float> vec1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> vec2 = {2.0, 3.0, 4.0, 5.0};
    std::vector<float> vec3 = {3.0, 4.0, 5.0, 6.0};
    appr_alg.addPoint((void*)vec1.data(), 1);
    appr_alg.addPoint((void*)vec2.data(), 2);
    appr_alg.addPoint((void*)vec3.data(), 3);
    // 删除向量
    appr_alg.markDelete(2);
    // 查询向量
    std::vector<float> query = {1.5, 2.5, 3.5, 4.5};
    auto result = appr_alg.searchKnn(query.data(), k);
    // 检查结果
    std::vector<int> expected_ids = {1, 3}; // 最近的两个点应该是vec1和vec3
    for (size_t i = 0; i < expected_ids.size(); i++) {
        auto res = result.top();
        result.pop();
        EXPECT_EQ(true, std::find(expected_ids.begin(), expected_ids.end(), res.second) != expected_ids.end());
    }
}

TEST(HNSWTest, MultiThreadLoad) {
    int d = 16;
    int max_elements = 100;

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;

    hnswlib::L2Space space(d);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, 2 * max_elements);

    int num_threads = 5;
    int num_labels = 10;

    int num_iterations = 10;
    int start_label = 0;

    while (true) {
        std::uniform_int_distribution<> distrib_int(start_label, start_label + num_labels - 1);
        std::vector<std::thread> threads;
        for (size_t thread_id = 0; thread_id < num_threads; thread_id++) {
            threads.push_back(
                std::thread(
                    [&] {
                        for (int iter = 0; iter < num_iterations; iter++) {
                            std::vector<float> data(d);
                            hnswlib::labeltype label = distrib_int(rng);
                            for (int i = 0; i < d; i++) {
                                data[i] = distrib_real(rng);
                            }
                            alg_hnsw->addPoint(data.data(), label);
                        }
                    }
                )
            );
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (alg_hnsw->cur_element_count > max_elements - num_labels) {
            break;
        }
        start_label += num_labels;
    }

    for (hnswlib::labeltype label = 0; label < max_elements; label++) {
        auto search = alg_hnsw->label_lookup_.find(label);
        if (search == alg_hnsw->label_lookup_.end()) {
            std::vector<float> data(d);
            for (int i = 0; i < d; i++) {
                data[i] = distrib_real(rng);
            }
            alg_hnsw->addPoint(data.data(), label);
        }
    }

    bool stop_threads = false;
    std::vector<std::thread> threads;

    num_threads = 20;
    int chunk_size = max_elements / num_threads;
    for (size_t thread_id = 0; thread_id < num_threads; thread_id++) {
        threads.push_back(
            std::thread(
                [&, thread_id] {
                    std::uniform_int_distribution<> distrib_int(0, chunk_size - 1);
                    int start_id = thread_id * chunk_size;
                    std::vector<bool> marked_deleted(chunk_size);
                    while (!stop_threads) {
                        int id = distrib_int(rng);
                        hnswlib::labeltype label = start_id + id;
                        if (marked_deleted[id]) {
                            alg_hnsw->unmarkDelete(label);
                            marked_deleted[id] = false;
                        } else {
                            alg_hnsw->markDelete(label);
                            marked_deleted[id] = true;
                        }
                    }
                }
            )
        );
    }

    num_threads = 20;
    std::uniform_int_distribution<> distrib_int_add(max_elements, 2 * max_elements - 1);
    for (size_t thread_id = 0; thread_id < num_threads; thread_id++) {
        threads.push_back(
            std::thread(
                [&] {
                    std::vector<float> data(d);
                    while (!stop_threads) {
                        hnswlib::labeltype label = distrib_int_add(rng);
                        for (int i = 0; i < d; i++) {
                            data[i] = distrib_real(rng);
                        }
                        alg_hnsw->addPoint(data.data(), label);
                        std::vector<float> data = alg_hnsw->getDataByLabel<float>(label);
                        float max_val = *max_element(data.begin(), data.end());
                        if (max_val > 10) {
                            throw std::runtime_error("Unexpected value in data");
                        }
                    }
                }
            )
        );
    }

    int sleep_ms = 1 * 1000;
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    stop_threads = true;
    for (auto &thread : threads) {
        thread.join();
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}