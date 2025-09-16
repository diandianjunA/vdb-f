#include "index_server/search_base.h"
#include "config/config.h"
#include "cuda/dist_calculate.cuh"
#include "lru/lru.h"
#include "rdma/rdma_pools.h"
#include <functional>
#include <future>
#include <queue>
#include <unordered_set>
#include <set>

// 计算距离
float calculate_distance(const float *a, const float *b, int dim) {
    // 使用openmp并行计算距离
    float distance = 0.0f;
#pragma omp parallel for reduction(+ : distance)
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        distance += diff * diff;
    }
    return distance;
}

std::pair<char *, std::shared_lock<std::shared_mutex>> get_data(int nodeid,
                                                                int shard) {
    int block_id = (nodeid / config->block_size) * config->shard_num + shard;
    int offset = nodeid % config->block_size;
    auto [data, read] = lru_->get_block(block_id);
    return {data + offset * config->size_data_per_element + config->offset_data,
            std::move(read)};
}

std::pair<char *, std::shared_lock<std::shared_mutex>> get_block(int nodeid,
                                                                 int shard) {
    int block_id = (nodeid / config->block_size) * config->shard_num + shard;
    auto [data, read] = lru_->get_block(block_id);
    return {data, std::move(read)};
}

void search(unsigned int entry_node, const float *query, int k, int ef_search,
            std::vector<long> &ids, std::vector<float> &distances, int shard) {
    int dim = config->dim;
    const std::vector<std::vector<int>> &neighbors = pools->get_neighbors(shard);

    std::priority_queue<Node, std::vector<Node>, std::less<Node>> top_k;
    std::multiset<Node> candidate_list;
    std::unordered_set<long> visited_nodes;
    DistanceCalculator distance_calculator(query, dim,
                                           config->size_data_per_element);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // 计算entry_node到query的距离
    auto [data, read] = get_data(entry_node, shard);
    float distance =
        distance_calculator.computeSingleDistanceAsync(data, stream);
    read.unlock();
    // 将entry_node加入候选列表
    candidate_list.insert(Node(entry_node, distance));
    // 将entry_node加入访问列表
    visited_nodes.insert(entry_node);
    while (!candidate_list.empty()) {
        Node node = *candidate_list.begin();
        candidate_list.erase(candidate_list.begin());
        if (top_k.size() < k) {
            top_k.push(node);
        } else if (node.distance < top_k.top().distance) {
            top_k.pop();
            top_k.push(node);
        }
        // 这里可以添加逻辑来获取邻居节点并推送到候选列表
        unsigned short int neighbor_count = neighbors[node.nodeid].size();
        std::vector<unsigned int> neighbors_id;
        std::vector<char *> neighbors_addr;
        std::vector<std::shared_lock<std::shared_mutex>> read_locks;
        for (int j = 0; j < neighbor_count; ++j) {
            int neighbor_id = neighbors[node.nodeid][j];
            if (visited_nodes.find(neighbor_id) != visited_nodes.end()) {
                continue;
            }
            neighbors_id.push_back(neighbor_id);
            auto [data, read] = get_data(neighbor_id, shard);
            neighbors_addr.push_back(data);
            // 这里使用std::move来避免不必要的拷贝
            read_locks.push_back(std::move(read));
            visited_nodes.insert(neighbor_id);
        }
        // 使用CUDA计算距离
        std::future<std::vector<float>> future_distances =
            std::async(std::launch::async, [&]() {
                return distance_calculator.computeDistances(neighbors_addr,
                                                            stream);
            });
        for (int j = 0; j < read_locks.size(); ++j) {
            read_locks[j].unlock();
        }
        std::vector<float> distances = future_distances.get();
        for (int j = 0; j < distances.size(); ++j) {
            float distance = distances[j];
            if (candidate_list.size() < ef_search) {
                candidate_list.insert(Node(neighbors_id[j], distance));
            } else if (distance < candidate_list.rbegin()->distance) {
                // 如果当前节点的距离小于候选列表中的最大距离，更新候选列表
                candidate_list.insert(Node(neighbors_id[j], distance));
                if (candidate_list.size() > ef_search) {
                    candidate_list.erase(--candidate_list.end());
                }
            }
        }
        // 如果当前候选列表中的最小距离大于top_k的最大距离，停止搜索
        if (candidate_list.begin()->distance > top_k.top().distance) {
            break;
        }
    }
    while (!top_k.empty()) {
        Node node = top_k.top();
        top_k.pop();
        ids.push_back(node.nodeid);
        distances.push_back(node.distance);
    }
    cudaStreamDestroy(stream);
}

// void search(unsigned int entry_node, const float *query, int k, int
// ef_search,
//             std::vector<long> &ids, std::vector<float> &distances, int shard)
//             {
//     int dim = config->dim;
//     int block_size = config->block_size;
//     int size_data_per_element = config->size_data_per_element;
//     const std::vector<std::vector<int>> &neighbors =
//     pool->get_neighbors(shard);

//     std::priority_queue<Node, std::vector<Node>, std::less<Node>> top_k;
//     std::multiset<Node> candidate_list;
//     std::unordered_map<long, float> visited_nodes;
//     DistanceCalculator distance_calculator(query, dim,
//     size_data_per_element); cudaStream_t stream; cudaStreamCreate(&stream);
//     // 计算entry_node到query的距离
//     auto [data, read] = get_data(entry_node, shard);
//     float distance =
//         distance_calculator.computeSingleDistanceAsync(data, stream);
//     read.unlock();
//     // 将entry_node加入候选列表
//     candidate_list.insert(Node(entry_node, distance));
//     // 将entry_node加入访问列表
//     visited_nodes[entry_node] = distance;
//     std::function<bool(int)> calculate_distance_func = [&](int nodeid) ->
//     bool {
//         auto [data, read] = get_block(nodeid, shard);
//         std::vector<float> distances =
//             distance_calculator.computeDistances(data, block_size, stream);
//         read.unlock();
//         int block_id = nodeid / block_size;
//         int primary_node = block_id * block_size;
//         // 将block_id的所有节点加入候选列表
//         for (int i = 0; i < block_size; ++i) {
//             int node_id = primary_node + i;
//             visited_nodes[node_id] = distances[i];
//             float distance = distances[i];
//             if (candidate_list.size() < ef_search) {
//                 candidate_list.insert(Node(node_id, distance));
//             } else if (distance < candidate_list.rbegin()->distance) {
//                 // 如果当前节点的距离小于候选列表中的最大距离，更新候选列表
//                 candidate_list.insert(Node(node_id, distance));
//                 if (candidate_list.size() > ef_search) {
//                     candidate_list.erase(--candidate_list.end());
//                 }
//             }
//         }
//         return true;
//     };
//     int iteration_count = 0;
//     auto start_time = std::chrono::high_resolution_clock::now();
//     while (!candidate_list.empty()) {
//         iteration_count++;
//         Node node = *candidate_list.begin();
//         candidate_list.erase(candidate_list.begin());
//         if (top_k.size() < k) {
//             top_k.push(node);
//         } else if (node.distance < top_k.top().distance) {
//             top_k.pop();
//             top_k.push(node);
//         }
//         // 这里可以添加逻辑来获取邻居节点并推送到候选列表
//         unsigned short int neighbor_count = neighbors[node.nodeid].size();
//         std::vector<std::future<bool>> futures;
//         for (int j = 0; j < neighbor_count; ++j) {
//             int neighbor_id = neighbors[node.nodeid][j];
//             if (visited_nodes.find(neighbor_id) != visited_nodes.end()) {
//                 continue;
//             }
//             std::future<bool> future = std::async(
//                 std::launch::async, calculate_distance_func, neighbor_id);
//             futures.push_back(std::move(future));
//         }
//         for (auto &future : futures) {
//             future.get();
//         }
//         // 如果当前候选列表中的最小距离大于top_k的最大距离，停止搜索
//         if (top_k.size() == k &&
//             candidate_list.begin()->distance > top_k.top().distance) {
//             break;
//         }
//     }
//     auto end_time = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed_time = end_time - start_time;
//     GlobalLogger->info("Elapsed time: {} seconds", elapsed_time.count());
//     GlobalLogger->info("Iteration count: {}", iteration_count);
//     while (!top_k.empty()) {
//         Node node = top_k.top();
//         top_k.pop();
//         ids.push_back(node.nodeid);
//         distances.push_back(node.distance);
//     }
//     cudaStreamDestroy(stream);
// }