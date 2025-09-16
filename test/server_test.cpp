#include <gtest/gtest.h>
#include "log/logger.h"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <vector>

using json = nlohmann::json;

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// 插入向量数据到服务器
void insert(const std::vector<std::vector<float>>& vectors, 
            const std::string& url = "http://localhost:3000/insert") {
    
    // 准备 JSON 负载
    json payload;
    payload["operation"] = "insert";
    payload["objects"] = json::array();
    
    // 添加向量对象
    for (size_t i = 0; i < vectors.size(); ++i) {
        json obj;
        obj["id"] = i;
        obj["vector"] = vectors[i];
        payload["objects"].push_back(obj);
    }
    
    // 将 JSON 转换为字符串
    std::string jsonStr = payload.dump();
    
    CURL* curl = curl_easy_init();
    if (!curl) {
        GlobalLogger->error("无法初始化CURL");
        throw std::runtime_error("无法初始化CURL");
        return;
    }
    
    // 设置响应缓冲区
    std::string responseBuffer;
    
    // 设置 CURL 选项
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonStr.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, jsonStr.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBuffer);
    
    // 设置 HTTP 头
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    
    // 执行请求
    CURLcode res = curl_easy_perform(curl);
    
    // 清理
    curl_slist_free_all(headers);
    
    if (res != CURLE_OK) {
        GlobalLogger->error("请求失败: {}", curl_easy_strerror(res));
        throw std::runtime_error("请求失败: " + std::string(curl_easy_strerror(res)));
    } else {
        // 获取 HTTP 状态码
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        
        if (http_code == 200) {
            GlobalLogger->info("成功插入向量");
        } else {
            GlobalLogger->error("插入向量失败。状态码: {}, 响应: {}", http_code, responseBuffer);
            throw std::runtime_error("插入向量失败。状态码: " + std::to_string(http_code) + ", 响应: " + responseBuffer);
        }
    }
    
    curl_easy_cleanup(curl);
}

void search(const std::vector<std::vector<float>>& vectors, 
            int top_k = 5, 
            const std::string& url = "http://localhost:4000/search") {
    
    // 准备 JSON 负载
    json payload;
    payload["operation"] = "search";
    payload["k"] = top_k;
    payload["objects"] = json::array();
    
    // 添加搜索向量
    for (const auto& vec : vectors) {
        json obj;
        obj["vector"] = vec;
        payload["objects"].push_back(obj);
    }
    
    // 将 JSON 转换为字符串
    std::string jsonStr = payload.dump();
    
    CURL* curl = curl_easy_init();
    if (!curl) {
        GlobalLogger->error("无法初始化CURL");
        throw std::runtime_error("无法初始化CURL");
        return;
    }
    
    // 设置响应缓冲区
    std::string responseBuffer;
    
    // 设置 CURL 选项
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonStr.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, jsonStr.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBuffer);
    
    // 设置 HTTP 头
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    
    // 执行请求
    CURLcode res = curl_easy_perform(curl);
    
    // 清理
    curl_slist_free_all(headers);
    
    if (res != CURLE_OK) {
        GlobalLogger->error("搜索请求失败: {}", curl_easy_strerror(res));
        throw std::runtime_error("搜索请求失败: " + std::string(curl_easy_strerror(res)));
    } else {
        // 获取 HTTP 状态码
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        
        if (http_code == 200) {
            try {
                // 解析 JSON 响应
                json result = json::parse(responseBuffer);
                GlobalLogger->info("搜索结果: {}", result.dump(4));
            } catch (const json::parse_error& e) {
                GlobalLogger->error("解析响应失败: {}", e.what());
                GlobalLogger->error("原始响应: {}", responseBuffer);
                throw std::runtime_error("解析响应失败: " + std::string(e.what()));
            }
        } else {
            GlobalLogger->error("搜索失败。状态码: {}, 响应: {}", http_code, responseBuffer);
            throw std::runtime_error("搜索失败。状态码: " + std::to_string(http_code) + ", 响应: " + responseBuffer);
        }
    }
    curl_easy_cleanup(curl);
}

void query(const std::vector<int>& ids, 
                  const std::string& url = "http://localhost:3000/query") {
    
    // 准备 JSON 负载
    json payload;
    payload["operation"] = "query";
    payload["ids"] = ids;
    
    // 将 JSON 转换为字符串
    std::string jsonStr = payload.dump();
    
    CURL* curl = curl_easy_init();
    if (!curl) {
        GlobalLogger->error("无法初始化CURL");
        throw std::runtime_error("无法初始化CURL");
    }
    
    // 设置响应缓冲区
    std::string responseBuffer;
    
    // 设置 CURL 选项
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonStr.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, jsonStr.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBuffer);
    
    // 设置 HTTP 头
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    
    // 执行请求
    CURLcode res = curl_easy_perform(curl);
    
    // 清理
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        GlobalLogger->error("查询请求失败: {}", curl_easy_strerror(res));
        throw std::runtime_error("查询请求失败: " + std::string(curl_easy_strerror(res)));
    }
    
    // 获取 HTTP 状态码
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    
    if (http_code != 200) {
        GlobalLogger->error("查询失败。状态码: {}, 响应: {}", http_code, responseBuffer);
        throw std::runtime_error("查询失败。状态码: " + std::to_string(http_code) + ", 响应: " + responseBuffer);
    }

    try {
        json result = json::parse(responseBuffer);
        GlobalLogger->info("查询结果: {}", result.dump(4));
    } catch (const json::parse_error& e) {
        GlobalLogger->error("解析响应失败: {}", e.what());
        GlobalLogger->error("原始响应: {}", responseBuffer);
        throw std::runtime_error("解析响应失败: " + std::string(e.what()));
    }
}

TEST(IndexHttpServerTest, Insert) {
    std::vector<std::vector<float>> vectors = {
        {1.0f, 2.0f, 3.0f, 4.0f}
    };
    EXPECT_NO_THROW(insert(vectors));
}

TEST(IndexHttpServerTest, Search) {
    std::vector<std::vector<float>> vectors = {
        {1.0f, 2.0f, 3.0f, 4.0f}
    };
    EXPECT_NO_THROW(search(vectors, 5));
}

TEST(IndexHttpServerTest, Query) {
    std::vector<int> ids = {0};
    EXPECT_NO_THROW(query(ids));
}

TEST(IndexHttpServerTest, MultiInsert) {
    std::vector<std::vector<float>> vectors = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {4.0f, 5.0f, 6.0f, 7.0f},
        {7.0f, 8.0f, 9.0f, 10.0f}
    };
    EXPECT_NO_THROW(insert(vectors));
}

TEST(IndexHttpServerTest, MultiSearch) {
    std::vector<std::vector<float>> vectors = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {4.0f, 5.0f, 6.0f, 7.0f},
        {7.0f, 8.0f, 9.0f, 10.0f}
    };
    EXPECT_NO_THROW(search(vectors, 5));
}

TEST(IndexHttpServerTest, MultiQuery) {
    std::vector<int> ids = {0, 1, 2};
    EXPECT_NO_THROW(query(ids));
}

// 启动程序并返回进程ID
pid_t start_program(const std::vector<std::string>& args) {
    // 准备参数数组（execvp 需要）
    std::vector<char*> argv;
    for (const auto& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr); // 必须以 nullptr 结尾

    pid_t pid = fork(); // 创建子进程
    
    if (pid == -1) {
        std::cerr << "创建子进程失败" << std::endl;
        return -1;
    } else if (pid == 0) {
        // 子进程：执行目标程序
        execvp(argv[0], argv.data());
        
        // 如果执行失败
        std::cerr << "执行程序失败: " << strerror(errno) << std::endl;
        _exit(1);
    }
    
    return pid; // 父进程返回子进程ID
}

// 停止指定进程
void stop_program(pid_t pid) {
    // 先尝试优雅终止
    if (kill(pid, SIGTERM) == -1) {
        std::cerr << "发送SIGTERM失败: " << strerror(errno) << std::endl;
        return;
    }
    
    // 等待程序退出（最多5秒）
    for (int i = 0; i < 5; ++i) {
        int status;
        pid_t result = waitpid(pid, &status, WNOHANG);
        
        if (result == pid) {
            std::cout << "程序已正常退出" << std::endl;
            return;
        } else if (result == -1) {
            std::cerr << "等待进程失败: " << strerror(errno) << std::endl;
            return;
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    // 强制终止
    std::cout << "程序未响应，发送SIGKILL强制终止" << std::endl;
    if (kill(pid, SIGKILL) == -1) {
        std::cerr << "发送SIGKILL失败: " << strerror(errno) << std::endl;
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    init_global_logger();
    set_log_level(spdlog::level::debug);

    std::vector<std::string> args1 = {"../../build/index_server", "../../config/conf5.ini"};
    pid_t pid1 = start_program(args1);
    if (pid1 == -1) {
        return 1;
    }

    std::vector<std::string> args2 = {"../../build/storage_server", "../../config/conf6.ini"};
    pid_t pid2 = start_program(args2);
    if (pid2 == -1) {
        return 1;
    }

    // 等待服务器启动
    std::this_thread::sleep_for(std::chrono::seconds(5));

    int ret = RUN_ALL_TESTS();

    stop_program(pid1);
    stop_program(pid2);

    return ret;
}