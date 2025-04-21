#include "storage_server/storage_http_server.h"
#include "rdma/rdma_pool.h"
#include "config/config.h"
#include "log/logger.h"

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
    StorageHttpServer server("0.0.0.0", 3000);
    GlobalLogger->info("Storage server created");
    GlobalLogger->info("Memory manager created");
    GlobalLogger->info("Buffer set");
    pool = std::make_shared<RdmaPool>(server.get_buffer(), server.get_size());
    GlobalLogger->info("RDMA pool created");
    GlobalLogger->info("Storage server started");
    server.start();
    return 0;
}