#pragma once
#include "config/config.h"
#include "log/logger.h"
#include <atomic>
#include <condition_variable>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <memory>

struct Frame {

    Frame(int frame_id) : frame_id(frame_id) {}

    Frame(int frame_id, char *data) : frame_id(frame_id), data(data) {}

    ~Frame() {}

    int frame_id;
    int block_id;
    char *data;
    std::atomic<int> use_count{0}; // 引用计数
};

class LRU {
  public:
    LRU(size_t capacity, char *buffer_);

    ~LRU() {}

    char *get_block(int block_id);
    void release_block(int block_id) {
        // std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = cache_map_.find(block_id);
        if (it != cache_map_.end()) {
            release_frame(it->second);
        }
    }
    void load_block_message(std::vector<char *> &data,
                            std::vector<int> &data_content);

  private:
    char *buffer_;
    std::mutex cache_mutex_;
    std::mutex lru_mutex_;
    std::condition_variable cv_;
    std::unordered_map<int, int> cache_map_;
    std::vector<Frame *> frame_map;
    std::list<int> lru_list_;
    size_t capacity_;

    void acquire_frame(int frame_id) {
        if (config->block_policy == BLOCK_POLICY::LOCKED) {
            frame_map[frame_id]->use_count.fetch_add(1,
                                                     std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(lru_mutex_);
            lru_list_.remove(frame_id); // 移除旧位置
        } else {
            std::lock_guard<std::mutex> lock(lru_mutex_);
            lru_list_.remove(frame_id); // 移除旧位置
            lru_list_.push_front(frame_id); // 添加到LRU列表尾部
        }
    }

    // 释放块时减少引用计数
    void release_frame(int frame_id) {
        if (config->block_policy == BLOCK_POLICY::LOCKED) {
            if (frame_map[frame_id]->use_count.fetch_sub(
                    1, std::memory_order_relaxed) == 1) {
                std::lock_guard<std::mutex> lock(lru_mutex_);
                lru_list_.remove(frame_id); // 移除旧位置
                lru_list_.push_front(frame_id); // 添m加到LRU列表尾部
            }
        }
    }
    // 安全驱逐逻辑
    int find_victim();

    void load_from_rdma(int block_id, char *addr);
};

extern std::shared_ptr<LRU> lru_;
