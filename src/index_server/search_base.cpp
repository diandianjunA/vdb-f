#include "index_server/search_base.h"
#include "config/config.h"
#include "cuda/dist_calculate.cuh"
#include "lru/lru.h"
#include "rdma/rdma_pools.h"
#include <functional>
#include <queue>
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

char *get_data(int nodeid, int shard) {
    auto start_time = std::chrono::high_resolution_clock::now();
    int block_id = (nodeid / config->block_size) * config->shard_num + shard;
    int offset = nodeid % config->block_size;
    char *data = lru_->get_block(block_id);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    get_block_time += elapsed_time;
    return data + offset * config->size_data_per_element + config->offset_data;
}

void search(unsigned int entry_node, const float *query, int k, int ef_search,
            std::vector<long> &ids, std::vector<float> &distances, int shard) {
    int dim = config->dim;

    const std::vector<std::vector<int>> &neighbors = pool->get_neighbors(shard);

    std::priority_queue<Node, std::vector<Node>, std::less<Node>> top_k;
    std::multiset<Node> candidate_list;
    std::set<long> visited_nodes;
    DistanceCalculator distance_calculator(query, dim);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // 计算entry_node到query的距离
    char *data = get_data(entry_node, shard);
    float distance =
        distance_calculator.computeSingleDistanceAsync(data, stream);
    cudaStreamSynchronize(stream);
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
        std::vector<unsigned int> neighbors_id(neighbor_count);
        std::vector<char *> neighbors_addr(neighbor_count);
        for (int j = 0; j < neighbor_count; ++j) {
            int neighbor_id = neighbors[node.nodeid][j];
            neighbors_id[j] = neighbor_id;
            neighbors_addr[j] = get_data(neighbor_id, shard);
        }
        // 使用CUDA计算距离
        std::vector<float> distances =
            distance_calculator.computeDistances(neighbors_addr, stream);
        cudaStreamSynchronize(stream);
        for (int j = 0; j < neighbor_count; ++j) {
            float distance = distances[j];
            if (visited_nodes.find(neighbors_id[j]) == visited_nodes.end()) {
                if (candidate_list.size() < ef_search) {
                    candidate_list.insert(Node(neighbors_id[j], distance));
                } else if (distance < candidate_list.rbegin()->distance) {
                    // 如果当前节点的距离小于候选列表中的最大距离，更新候选列表
                    candidate_list.insert(Node(neighbors_id[j], distance));
                    if (candidate_list.size() > ef_search) {
                        candidate_list.erase(--candidate_list.end());
                    }
                }
                // 记录访问过的节点
                visited_nodes.insert(neighbors_id[j]);
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