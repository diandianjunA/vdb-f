#pragma once
#include <string>
#include <memory>
#include <chrono>

extern int get_total;
extern int get_hit;
extern int rdma_request;
extern std::chrono::duration<double> rdma_elpased_time;
extern std::chrono::duration<double> get_block_time;

enum BLOCK_POLICY {
    LOCKED,
    UNLOCKED,
};

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