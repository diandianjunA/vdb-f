#include "index_server/index_engine.h"
#include "config/config.h"
#include "index_server/search_base.h"
#include "log/logger.h"
#include "rdma/rdma_pools.h"
#include <chrono>
#include <functional>
#include <map>
#include <queue>
#include <thread>

IndexEngine::IndexEngine(int dim) : dim(dim) {}

// 查询向量
std::pair<std::vector<long>, std::vector<float>>
IndexEngine::search_vectors(const std::vector<float> &query, int k,
                            int ef_search) {
    int query_num = query.size() / dim;
    std::vector<long> result_ids(query_num * k);
    std::vector<float> result_distances(query_num * k,
                                        std::numeric_limits<float>::max());

    int shard_num = config->shard_num;
    std::vector<std::thread> threads;
    std::vector<std::priority_queue<std::pair<float, long>,
                                    std::vector<std::pair<float, long>>,
                                    std::less<std::pair<float, long>>>>
        result_queue(query_num);
    std::vector<std::mutex> queue_mutex(query_num);

    for (int i = 0; i < shard_num; i++) {
        threads.push_back(std::thread([&, i]() {
            unsigned int entry_node = pool->get_entry_node(i);
            ParallelFor(0, query_num, 0, [&](size_t index, size_t threadId) {
                std::vector<long> ids;
                std::vector<float> distances;
                auto start_time = std::chrono::high_resolution_clock::now();
                search(entry_node, query.data() + index * dim, k, ef_search,
                       ids, distances, i);
                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_time =
                    end_time - start_time;
                GlobalLogger->info("Thread {}: Search time: {} seconds",
                                   threadId, elapsed_time.count());
                GlobalLogger->info("get hit: {}, get total: {}",
                                   get_hit, get_total);
                GlobalLogger->info("rdma request: {}, rdma elapsed time: {}",
                                   rdma_request,
                                   rdma_elpased_time.count());
                GlobalLogger->info("get block time: {}", get_block_time.count());
                for (int j = 0; j < k; j++) {
                    std::lock_guard<std::mutex> lock(queue_mutex[index]);
                    result_queue[index].push(
                        std::make_pair(distances[j], ids[j]));
                    // 如果队列大小超过k，弹出最远的元素
                    if (result_queue[index].size() > k) {
                        result_queue[index].pop();
                    }
                }
            });
        }));
    }
    for (auto &thread : threads) {
        thread.join();
    }
    for (int i = 0; i < query_num; i++) {
        for (int j = 0; j < k; j++) {
            result_ids[i * k + j] = result_queue[i].top().second;
            result_distances[i * k + j] = result_queue[i].top().first;
            result_queue[i].pop();
        }
    }
    return {result_ids, result_distances};
}
