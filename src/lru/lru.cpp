#include "lru/lru.h"
#include "config/config.h"
#include "rdma/rdma_pools.h"

std::shared_ptr<LRU> lru_;

LRU::LRU(size_t capacity, char *buffer_)
    : capacity_(capacity), buffer_(buffer_) {
    frame_map.reserve(capacity);
    // 初始化LRU缓存
    for (size_t i = 0; i < capacity; ++i) {
        Frame *frame = new Frame(i);
        frame->data =
            buffer_ + i * config->block_size * config->size_data_per_element;
        frame->use_count.store(0, std::memory_order_relaxed);
        frame->block_id = -1;
        frame_map.push_back(frame);
        lru_list_.push_back(i);
    }
}

// 安全驱逐逻辑
int LRU::find_victim() {
    std::lock_guard<std::mutex> lock(lru_mutex_);
    if (lru_list_.empty()) {
        return -1; // 没有可驱逐的块
    }
    int victim = lru_list_.back();
    lru_list_.pop_back();
    return victim;
}

char *LRU::get_block(int block_id) {
    get_total++;

    while (true) {
        auto it = cache_map_.find(block_id);
        if (it != cache_map_.end()) {
            get_hit++;
            int frame_id = it->second;
            acquire_frame(frame_id);
            return frame_map[frame_id]->data;
        } else {
            int victim = find_victim();
            if (victim == -1) {
                return nullptr; // 没有可驱逐的块
            }
            Frame *frame = frame_map[victim];
            if (cache_map_.find(frame->block_id) != cache_map_.end()) {
                std::unique_lock<std::mutex> lock(cache_mutex_);
                cache_map_.erase(frame->block_id);
            }
            frame->block_id = block_id;
            std::unique_lock<std::mutex> lock(cache_mutex_);
            cache_map_[block_id] = victim;

            // 释放锁进行IO
            // lock.unlock();
            load_from_rdma(block_id, frame->data);
            // lock.lock();

            acquire_frame(victim);

            cv_.notify_all();
            return frame->data;
        }
    }
}

void LRU::load_from_rdma(int block_id, char *addr) {
    int qp_id = block_id % config->shard_num;
    int inner_block_id = block_id / config->shard_num;
    int primary_node = inner_block_id * config->block_size;
    int size = config->block_size * config->size_data_per_element;
    pool.get()->rdma_read(qp_id, addr, size,
                          primary_node * config->size_data_per_element);
}

void LRU::load_block_message(std::vector<char *> &data,
                             std::vector<int> &data_content) {
    for (auto &pair : cache_map_) {
        int block_id = pair.first;
        int frame_id = pair.second;
        data.push_back(frame_map[frame_id]->data);
        data_content.push_back(block_id);
    }
}