#include "config/config.h"
#include "index_server/index_http_client.h"
#include "index_server/index_http_server.h"
#include "log/logger.h"
#include "lru/lru.h"
#include "rdma/rdma_pools.h"
#include <cuda_runtime.h>

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    // 从命令行参数读取配置文件路径
    std::string config_file_path = argv[1];

    init_global_logger();
    set_log_level(spdlog::level::debug);
    GlobalLogger->info("Global logger initialized");

    // 读取配置文件
    std::map<std::string, std::string> f_config = readConfig(config_file_path);

    device = std::make_shared<RdmaDevice>();
    config->dim = std::stoi(f_config["dim"]);
    config->num_data = std::stoi(f_config["num_data"]);
    config->max_m = std::stoi(f_config["max_m"]);
    config->ef_construction = std::stoi(f_config["ef_construction"]);
    config->shard_num = 1;
    config->memory_region_size = 1000 * 1024 * 1024;
    config->block_size = 100;
    config->block_policy = BLOCK_POLICY::UNLOCKED;
    // config->block_policy = BLOCK_POLICY::LOCKED;
    config->calculate();
    // std::string etcdEndpoints = "http://127.0.0.1:2379";
    void *buf;
    cudaMalloc(&buf, config->memory_region_size);
    cudaMemset(buf, 0, config->memory_region_size);
    pools = std::make_shared<RdmaPools>(config->shard_num, buf,
                                       config->memory_region_size);
    auto load_from_rdma = [&](int block_id, char *addr) {
        int qp_id = block_id % config->shard_num;
        int inner_block_id = block_id / config->shard_num;
        int primary_node = inner_block_id * config->block_size;
        int size = config->block_size * config->size_data_per_element;
        pools->rdma_read(qp_id, addr, size,
                        primary_node * config->size_data_per_element);
    };
    lru_ = std::make_shared<LRU>(config->num_block, (char *)buf, load_from_rdma);
    GlobalLogger->info("RDMA pool created");
    GlobalLogger->info("LRU cache created");

    // etcdClient_ = std::make_unique<etcd::Client>(etcdEndpoints);
    int http_server_port = std::stoi(f_config["http_server_port"]);
    IndexHttpServer indexHttpServer("0.0.0.0", http_server_port);
    GlobalLogger->info("Index server started");
    indexHttpServer.start();

    return 0;
}