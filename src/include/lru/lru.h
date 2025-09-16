#pragma once
#include "config/config.h"
#include "log/logger.h"
#include <atomic>
#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

struct Frame {

    Frame(int frame_id) : frame_id(frame_id) {}

    Frame(int frame_id, char *data) : frame_id(frame_id), data(data) {}

    ~Frame() {}

    // 获取读锁（RAII）
    [[nodiscard]] auto acquire_read() {
        return std::shared_lock<std::shared_mutex>(mutex);
    }

    // 获取写锁（RAII）
    [[nodiscard]] auto acquire_write() {
        return std::unique_lock<std::shared_mutex>(mutex);
    }

    [[nodiscard]] auto try_acquire_write() {
        return std::unique_lock<std::shared_mutex>(mutex, std::try_to_lock);
    }

    int frame_id;
    int block_id;
    char *data;
    std::shared_mutex mutex;
};

class LRU {
  public:
    LRU(size_t capacity, char *buffer_, std::function<void(int, char *)> load, int element_per_shard = 30);

    ~LRU() {}

    std::pair<char *, std::shared_lock<std::shared_mutex>>
    get_block(int block_id);

  private:
    struct Shard {
        std::mutex mutex;
        size_t capacity;
        std::list<int> lru_list;
        std::unordered_map<int, std::list<int>::iterator> cache_map;

        Shard() : capacity(0) {}
        Shard(size_t capacity) : capacity(capacity) {}
        Shard(const Shard &other) = delete;
        Shard(Shard &&other) {
            capacity = other.capacity;
            lru_list = std::move(other.lru_list);
            cache_map = std::move(other.cache_map);
        }
        ~Shard() {}
    };

    std::vector<Shard> shards_;
    Shard &GetShard(const int &key) {
        static thread_local std::hash<int> hash;
        size_t hash_value = hash(key);
        return shards_[hash_value % shards_.size()];
    }

    char *buffer_;
    std::vector<Frame *> frame_map;
    size_t capacity_;
    std::function<void(int, char *)> load_;
};

extern std::shared_ptr<LRU> lru_;