#pragma once
#include <string>
#include <memory>
#include <chrono>
#include <map>
#include "log/logger.h"

enum BLOCK_POLICY {
    LOCKED,
    UNLOCKED,
};

std::map<std::string, std::string> readConfig(const std::string& filename);

class Config {

public:
    Config() {};
    int init(const std::string &config_file);
    void calculate();
    ~Config() {};
    int dim;
    int shard_num;
    int memory_region_size;
    int num_data;
    int max_m;
    int ef_construction;
    int size_data_per_element;
    int offset_data;
    int data_size;
    int block_size;
    int num_block;
    enum BLOCK_POLICY block_policy;
};

extern std::shared_ptr<Config> config;