#include "lru/lru.h"
#include "config/config.h"

std::shared_ptr<LRU> lru_;

LRU::LRU(size_t capacity, char *buffer_, std::function<void(int, char *)> load, int element_per_shard)
    : capacity_(capacity), buffer_(buffer_), load_(load),
      shards_(capacity / element_per_shard) {
    frame_map.reserve(capacity);
    // 初始化LRU缓存
    for (size_t i = 0; i < capacity; ++i) {
        Frame *frame = new Frame(i);
        frame->data =
            buffer_ + i * config->block_size * config->size_data_per_element;
        frame->block_id = -1;
        frame_map.push_back(frame);
    }
    if (capacity < element_per_shard) {
        element_per_shard = capacity;
        shards_.push_back(Shard(capacity));
        for (size_t j = 0; j < shards_[0].capacity; ++j) {
            shards_[0].lru_list.push_back(j);
            shards_[0].cache_map[j] = shards_[0].lru_list.end();
            shards_[0].cache_map[j]--;
        }
    } else {
        int shard_count = capacity / element_per_shard;
        size_t per_shard_base = capacity / shard_count;
        size_t remainder = capacity % shard_count;
        for (size_t i = 0; i < shard_count; ++i) {
            shards_[i].capacity = per_shard_base + (i < remainder ? 1 : 0);
            for (size_t j = 0; j < shards_[i].capacity; ++j) {
                int frame_id = i * per_shard_base + j;
                if (i < remainder) {
                    frame_id += i;
                } else {
                    frame_id += remainder;
                }
                shards_[i].lru_list.push_back(frame_id);
                shards_[i].cache_map[frame_id] = shards_[i].lru_list.end();
                shards_[i].cache_map[frame_id]--;
            }
        }
    }
}

std::pair<char *, std::shared_lock<std::shared_mutex>>
LRU::get_block(int block_id) {
    auto &shard = GetShard(block_id);
    // 尝试获取锁
    std::unique_lock<std::mutex> lock(shard.mutex);
    auto it = shard.cache_map.find(block_id);
    if (it == shard.cache_map.end() || frame_map[*(it->second)]->block_id != block_id) {
        int frame_id = shard.lru_list.back();
        Frame *frame = frame_map[frame_id];
        std::unique_lock<std::shared_mutex> write;
        write = frame->acquire_write();
        if (shard.cache_map.find(frame->block_id) != shard.cache_map.end()) {
            std::list<int>::iterator it = shard.cache_map[frame->block_id];
            shard.lru_list.splice(shard.lru_list.begin(), shard.lru_list, it);
            shard.cache_map.erase(frame->block_id);
        } else {
            auto last = shard.lru_list.end();
            --last; // 获取最后一个元素的迭代器
            shard.lru_list.splice(shard.lru_list.begin(), shard.lru_list, last);
        }
        frame->block_id = block_id;
        load_(block_id, frame->data);
        shard.cache_map[block_id] = shard.lru_list.begin();
        write.unlock();
        auto read = frame->acquire_read();
        return {frame->data, std::move(read)};
    } else {
        // 将访问的节点移到链表头部
        shard.lru_list.splice(shard.lru_list.begin(), shard.lru_list,
                              it->second);
        int frame_id = *(it->second);
        auto read = frame_map[frame_id]->acquire_read();
        return {frame_map[frame_id]->data, std::move(read)};
    }
}