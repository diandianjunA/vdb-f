#include <gtest/gtest.h>
#include "lru/lru.h"
#include "config/config.h"
#include "log/logger.h"
#include <iostream>
#include <string>

TEST(LRU, GetBlock) {
    config->dim = 128;
    config->max_m = 32;
    config->memory_region_size = 1024 * 1024;
    config->block_size = 10;
    config->calculate();

    size_t buf_size = config->memory_region_size;
    char *buf = (char *)malloc(buf_size);
    auto load = [&](int block_id, char *addr) {
        std::string block = std::string("block") + std::to_string(block_id);
        memcpy(addr, block.c_str(), config->block_size);
    };
    LRU lru(10, (char *)buf, load);
    auto [data1, lock1] = lru.get_block(1);
    auto [data2, lock2] = lru.get_block(2);
    
    EXPECT_EQ(strcmp(data1, "block1"), 0);
    EXPECT_EQ(strcmp(data2, "block2"), 0);
    auto [data1_again, lock1_again] = lru.get_block(1);
    EXPECT_EQ(strcmp(data1_again, "block1"), 0);
    EXPECT_EQ(strcmp(data2, "block2"), 0);
}

TEST(LRU, Eviction) {
    config->dim = 128;
    config->max_m = 32;
    config->memory_region_size = 1024 * 1024;
    config->block_size = 10;
    config->calculate();

    size_t buf_size = config->memory_region_size;
    char *buf = (char *)malloc(buf_size);
    auto load = [&](int block_id, char *addr) {
        std::string block = std::string("block") + std::to_string(block_id);
        memcpy(addr, block.c_str(), config->block_size);
    };
    LRU lru(3, (char *)buf, load);
    lru.get_block(1);
    lru.get_block(2);
    lru.get_block(3);
    // 此时缓存已满，访问新块应触发淘汰
    lru.get_block(4);
    auto [data1, lock1] = lru.get_block(1); // 块1应被淘汰，需重新加载
    std::cout << "data1: " << data1 << std::endl;
    EXPECT_EQ(strcmp(data1, "block1"), 0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    init_global_logger();
    set_log_level(spdlog::level::debug);
    return RUN_ALL_TESTS();
}