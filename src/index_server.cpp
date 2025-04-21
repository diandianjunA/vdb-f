#include "config/config.h"
#include "index_server/index_http_client.h"
#include "index_server/index_http_server.h"
#include "log/logger.h"
#include "rdma/rdma_pools.h"
#include "lru/lru.h"
#include <cuda_runtime.h>

int main() {

    init_global_logger();
    set_log_level(spdlog::level::debug);
    GlobalLogger->info("Global logger initialized");

    device = std::make_shared<RdmaDevice>();
    int num_data = 100000;
    int dim = 128;
    int max_m = 32;
    int ef_construction = 200;
    config->dim = dim;
    config->num_data = num_data;
    config->max_m = max_m;
    config->ef_construction = ef_construction;
    config->shard_num = 1;
    config->memory_region_size = 100 * 1024 * 1024;
    config->block_size = 100;
    config->block_policy = BLOCK_POLICY::UNLOCKED;
    // config->block_policy = BLOCK_POLICY::LOCKED;
    config->calculate();
    // std::string etcdEndpoints = "http://127.0.0.1:2379";
    void* buf;
    cudaMalloc(&buf, config->memory_region_size);
    cudaMemset(buf, 0, config->memory_region_size);
    pool = std::make_shared<RdmaPools>(config->shard_num,
                                       buf,
                                       config->memory_region_size);
    lru_ = std::make_shared<LRU>(config->num_block, (char*)buf);
    GlobalLogger->info("RDMA pool created");
    GlobalLogger->info("LRU cache created");

    // etcdClient_ = std::make_unique<etcd::Client>(etcdEndpoints);
    IndexHttpServer indexHttpServer("0.0.0.0", 4000);
    GlobalLogger->info("Index server started");
    indexHttpServer.start();

    return 0;
}