#include "config/config.h"
#include "log/logger.h"
#include "rdma/rdma_pool.h"
#include "storage_server/storage_http_server.h"
#include <filesystem>

namespace fs = std::filesystem;

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
    auto f_config = readConfig(config_file_path);
    device = std::make_shared<RdmaDevice>();
    int num_data = std::stoi(f_config["num_data"]);
    int dim = std::stoi(f_config["dim"]);
    int max_m = std::stoi(f_config["max_m"]);
    int ef_construction = std::stoi(f_config["ef_construction"]);
    config->dim = dim;
    config->num_data = num_data;
    config->max_m = max_m;
    config->ef_construction = ef_construction;
    config->shard_num = 1;

    auto reset_directory = [](const fs::path &dir_path) {
        try {
            // 如果目标文件夹已存在
            if (fs::exists(dir_path)) {
                if (!fs::is_directory(dir_path)) {
                    GlobalLogger->error("{} 不是一个文件夹", dir_path.string());
                    throw std::runtime_error(dir_path.string() + " 不是一个文件夹");
                }
                // 清空文件夹内容
                fs::remove_all(dir_path);
            }
            // 重新创建空文件夹
            fs::create_directories(dir_path);
        } catch (const std::exception &e) {
            GlobalLogger->error("操作失败: {}", e.what());
            throw std::runtime_error(e.what());
        }
    };

    std::string base_path = f_config["base_path"];
    reset_directory(base_path);
    std::string db_path = base_path + "/rocksdb";

    int http_server_port = std::stoi(f_config["http_server_port"]);

    StorageHttpServer server("0.0.0.0", http_server_port, db_path);
    GlobalLogger->info("Storage server created");
    GlobalLogger->info("Memory manager created");
    GlobalLogger->info("Buffer set");
    pool = std::make_shared<RdmaPool>(server.get_buffer(), server.get_size());
    GlobalLogger->info("RDMA pool created");
    GlobalLogger->info("Storage server started");
    server.start();
    return 0;
}