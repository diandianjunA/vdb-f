#include <gtest/gtest.h>
#include "config/config.h"

TEST(ConfigTest, ReadConfig) {
    init_global_logger();
    set_log_level(spdlog::level::debug);

    std::string filename = "/home/xjs/vdb-f/config/conf1.ini";
    auto config_map = readConfig(filename);
    EXPECT_EQ(config_map["http_server_port"], "4000");
    EXPECT_EQ(config_map["num_data"], "10000");
    EXPECT_EQ(config_map["dim"], "128");
    EXPECT_EQ(config_map["max_m"], "32");
    EXPECT_EQ(config_map["ef_construction"], "200");
}

TEST(ConfigTest, Calculate) {
    config->dim = 128;
    config->max_m = 32;
    config->memory_region_size = 1024 * 1024 * 1024; // 1GB
    config->block_size = 1000;
    config->calculate();
    EXPECT_EQ(config->data_size, 128 * sizeof(float));
    EXPECT_EQ(config->offset_data, 4 + 32 * sizeof(unsigned int));
    EXPECT_EQ(config->size_data_per_element,
              config->offset_data + config->data_size + sizeof(size_t));
    EXPECT_EQ(config->num_block,
              config->memory_region_size / (config->block_size * config->size_data_per_element));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}