#include "storage_server/storage_http_server.h"
#include "config/config.h"
#include "log/logger.h"
#include "rdma/rdma_common.h"
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <thread>

template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads,
                        Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(
                            lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value
                         * that size_t can fit, because fetch_add returns the
                         * previous value before the increment (what will result
                         * in overflow and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

std::vector<float> randVecs(size_t num, size_t dim) {
    std::vector<float> v(num * dim);
    // 生成指定数量的随机向量
    for (size_t i = 0; i < num; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            v[i * dim + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    return v;
}

StorageHttpServer::StorageHttpServer(const std::string &host, int port,
                                     const std::string &db_path)
    : host(host), port(port) {
    server.Post("/connectRdma",
                [this](const httplib::Request &req, httplib::Response &res) {
                    connectRdma(req, res);
                });
    server.Post("/insert",
                [this](const httplib::Request &req, httplib::Response &res) {
                    insertHandler(req, res);
                });
    server.Post("/query",
                [this](const httplib::Request &req, httplib::Response &res) {
                    queryHandler(req, res);
                });

    space = new hnswlib::L2Space(config->dim);
    index = new hnswlib::HierarchicalNSW<float>(
        space, config->num_data, config->max_m / 2, config->ef_construction);
    neighbor = new Neighbor(config->num_data);

    ParallelFor(0, config->num_data, 0, [&](size_t id, size_t threadId) {
        index->addPoint(randVecs(1, config->dim).data(), id);
    });

    rocksdb::DB *db;
    rocksdb::Options options;
    options.create_if_missing = true;
    rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db);
    if (!status.ok()) {
        throw std::runtime_error("rocksdb open error");
    }
    db_ = db;
}

void StorageHttpServer::start() { server.listen(host.c_str(), port); }

void setJsonResponse(const rapidjson::Document &json_response,
                     httplib::Response &res) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    json_response.Accept(writer);
    res.set_content(buffer.GetString(), "application/json");
    res.status = 200;
}

void setErrorJsonResponse(httplib::Response &res, int error_code,
                          const std::string &errorMsg) {
    rapidjson::Document json_response;
    json_response.SetObject();
    rapidjson::Document::AllocatorType &allocator =
        json_response.GetAllocator();
    json_response.AddMember("code", error_code, allocator);
    json_response.AddMember("error_msg", rapidjson::StringRef(errorMsg.c_str()),
                            allocator);
    setJsonResponse(json_response, res);
}

void StorageHttpServer::connectRdma(const httplib::Request &req,
                                    httplib::Response &res) {
    GlobalLogger->debug("Received RDMA connection request");
    // 解析JSON请求
    rapidjson::Document json_request;
    json_request.Parse(req.body.c_str());

    if (!json_request.IsObject()) {
        GlobalLogger->error("invalid JSON request");
        res.status = 400;
        setErrorJsonResponse(res, 400, "invalid JSON request");
        return;
    }

    uint64_t addr = json_request["addr"].GetUint64();
    uint32_t rkey = json_request["rkey"].GetUint();
    uint32_t qp_num = json_request["qp_num"].GetUint();
    uint16_t lid = json_request["lid"].GetUint();
    GlobalLogger->debug(
        "Received RDMA connection request: addr = {:#x}, rkey = "
        "{:#x}, qp_num = {:#x}, lid = {:#x}",
        addr, rkey, qp_num, lid);

    // 构造QP属性
    cm_con_data_t remote_props;
    remote_props.addr = addr;
    remote_props.rkey = rkey;
    remote_props.qp_num = qp_num;
    remote_props.lid = lid;

    QPEntry *qp = pool->add_qp();
    cm_con_data_t local_props;
    local_props.addr = (uint64_t)pool->get_buffer()->get_buf();
    local_props.rkey = (uint32_t)pool->get_buffer()->get_mr()->rkey;
    local_props.qp_num = qp->get_qp()->qp_num;
    local_props.lid = device->get_lid();

    GlobalLogger->debug(
        "Local RDMA connection properties: addr = {:#x}, rkey = "
        "{:#x}, qp_num = {:#x}, lid = {:#x}",
        local_props.addr, local_props.rkey, local_props.qp_num,
        local_props.lid);

    GlobalLogger->debug("size_data_per_element: {}",
                        index->size_data_per_element_);
    GlobalLogger->debug("offset_data: {}", index->offsetData_);
    GlobalLogger->debug("max_m: {}", index->maxM0_);
    GlobalLogger->debug("num_data: {}", index->max_elements_);
    GlobalLogger->debug("data_size: {}", index->data_size_);

    neighbor->update_neighbors(index);
    GlobalLogger->debug("Neighbor updated");
    std::string neighbor_json = neighbor->serialize();

    rapidjson::Document json_response(rapidjson::kObjectType);
    // json_response.SetObject();
    rapidjson::Document::AllocatorType &allocator =
        json_response.GetAllocator();
    json_response.AddMember("addr", local_props.addr, allocator);
    json_response.AddMember("rkey", local_props.rkey, allocator);
    json_response.AddMember("qp_num", local_props.qp_num, allocator);
    json_response.AddMember("lid", local_props.lid, allocator);
    json_response.AddMember("entry_node", index->enterpoint_node_, allocator);
    json_response.AddMember(
        "neighbor", rapidjson::StringRef(neighbor_json.c_str()), allocator);
    setJsonResponse(json_response, res);

    GlobalLogger->debug("send back");

    std::thread([qp, remote_props]() {
        // printf("Connecting QP to remote QP number %d\n",
        // remote_props.qp_num);
        qp->connect_qp(remote_props);
    }).detach();
}

void StorageHttpServer::insertHandler(const httplib::Request &req,
                                      httplib::Response &res) {
    GlobalLogger->debug("Received insert request");

    rapidjson::Document json_request;
    json_request.Parse(req.body.c_str());

    // 检查JSON文档是否为有效对象
    if (!json_request.IsObject()) {
        GlobalLogger->error("Invalid JSON request");
        res.status = 400;
        setErrorJsonResponse(res, 400, "Invalid JSON request");
        return;
    }

    const rapidjson::Value &objects = json_request["objects"];

    if (!objects.IsArray()) {
        throw std::runtime_error("objects type not match");
    }
    std::vector<float> vectors;
    std::vector<long> ids;

    for (auto &obj : objects.GetArray()) {
        if (obj.HasMember("vector") && obj["vector"].IsArray() &&
            obj.HasMember("id") && obj["id"].IsInt()) {
            const rapidjson::Value &row = obj["vector"];
            int id = obj["id"].GetInt();
            std::vector<float> vector;
            for (rapidjson::SizeType j = 0; j < row.Size(); j++) {
                vector.push_back(row[j].GetFloat());
            }
            vectors.insert(vectors.end(), vector.begin(), vector.end());
            ids.push_back(id);
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            row.Accept(writer);
            std::string value = buffer.GetString();
            db_->Put(rocksdb::WriteOptions(), std::to_string(id), value);
        } else {
            throw std::runtime_error(
                "Missing vectors or id parameter in the request");
        }
    }

    if (vectors.size() % config->dim != 0) {
        GlobalLogger->error("Invalid vector size");
        res.status = 400;
        setErrorJsonResponse(res, 400, "Invalid vector size");
        return;
    }

    if (vectors.size() / config->dim != ids.size()) {
        GlobalLogger->error("Invalid vector size");
        res.status = 400;
        setErrorJsonResponse(res, 400, "Invalid vector size");
        return;
    }

    ParallelFor(0, ids.size(), 0, [&](size_t i, size_t threadId) {
        index->addPoint(vectors.data() + i * config->dim, ids[i]);
    });

    rapidjson::Document json_response;
    json_response.SetObject();
    rapidjson::Document::AllocatorType &allocator =
        json_response.GetAllocator();

    // 添加retCode到响应
    json_response.AddMember("code", 0, allocator);

    setJsonResponse(json_response, res);
}

// 请求给出id，返回rocksdb中存储的数据
void StorageHttpServer::queryHandler(const httplib::Request& req, httplib::Response& res) {
    GlobalLogger->debug("Received query request");

    rapidjson::Document json_request;
    json_request.Parse(req.body.c_str());

    // 检查JSON文档是否为有效对象
    if (!json_request.IsObject()) {
        GlobalLogger->error("Invalid JSON request");
        res.status = 400;
        setErrorJsonResponse(res, 400, "Invalid JSON request");
        return;
    }

    const rapidjson::Value &ids = json_request["ids"];

    if (!ids.IsArray()) {
        throw std::runtime_error("ids type not match");
    }

    std::vector<int> id_list;
    for (auto &id : ids.GetArray()) {
        id_list.push_back(id.GetInt());
    }

    std::vector<std::string> values;
    for (auto &id : id_list) {
        std::string value;
        db_->Get(rocksdb::ReadOptions(), std::to_string(id), &value);
        values.push_back(value);
    }

    rapidjson::Document json_response;
    json_response.SetObject();
    rapidjson::Document::AllocatorType &allocator =
        json_response.GetAllocator();
    json_response.AddMember("values", rapidjson::StringRef(values[0].c_str()), allocator);
    setJsonResponse(json_response, res);
}
