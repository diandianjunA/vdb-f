#include "index_server/index_http_client.h"
#include "log/logger.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include <tuple>

std::unique_ptr<etcd::Client> etcdClient_;

std::vector<std::vector<int>> deserialize_vec2d(const std::string &json_str) {
    rapidjson::Document doc;
    doc.Parse(json_str.c_str());

    std::vector<std::vector<int>> result;
    for (auto &outer_value : doc.GetArray()) {
        std::vector<int> inner_vec;
        for (auto &inner_value : outer_value.GetArray()) {
            inner_vec.push_back(inner_value.GetInt());
        }
        result.push_back(inner_vec);
    }
    return result;
}

std::tuple<struct cm_con_data_t, unsigned int, std::vector<std::vector<int>>>
IndexHttpClient::connectRdma(uint64_t addr, uint32_t rkey, uint32_t qp_num,
                             uint16_t lid) {
    client.set_read_timeout(30, 0);
    client.set_connection_timeout(10, 0);
    rapidjson::Document json_request;
    json_request.SetObject();
    rapidjson::Document::AllocatorType &allocator = json_request.GetAllocator();
    json_request.AddMember("addr", addr, allocator);
    json_request.AddMember("rkey", rkey, allocator);
    json_request.AddMember("qp_num", qp_num, allocator);
    json_request.AddMember("lid", lid, allocator);

    GlobalLogger->debug("Sending RDMA connection request: addr = {:#x}, rkey = "
                        "{:#x}, qp_num = {:#x}, lid = {:#x}",
                        addr, rkey, qp_num, lid);

    httplib::Headers headers = {
        {"Content-Type", "application/json"},
    };
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    json_request.Accept(writer);
    auto res = client.Post("/connectRdma", headers, buffer.GetString(),
                           "application/json");
    if (!res) {
        throw std::runtime_error("Failed to connect to server");
    }
    if (res->status != 200) {
        throw std::runtime_error("Failed to connect to server");
    }

    rapidjson::Document json_response;
    json_response.Parse(res->body.c_str());
    if (!json_response.IsObject()) {
        throw std::runtime_error("Invalid JSON response");
    }

    cm_con_data_t remote_props;
    remote_props.addr = json_response["addr"].GetUint64();
    remote_props.rkey = json_response["rkey"].GetUint();
    remote_props.qp_num = json_response["qp_num"].GetUint();
    remote_props.lid = json_response["lid"].GetUint();
    unsigned int entry_id = json_response["entry_node"].GetUint();
    std::string neighbor_json = json_response["neighbor"].GetString();
    std::vector<std::vector<int>> neighbors =
        deserialize_vec2d(neighbor_json);

    GlobalLogger->debug("Received RDMA connection response: addr = {:#x}, rkey "
                        "= {:#x}, qp_num = {:#x}, lid = {:#x}",
                        remote_props.addr, remote_props.rkey,
                        remote_props.qp_num, remote_props.lid);
    GlobalLogger->debug("Entry node: {}", entry_id);

    return {remote_props, entry_id, neighbors};
}
