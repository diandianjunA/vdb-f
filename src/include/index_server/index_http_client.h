#pragma once
#include "httplib.h"
#include "rapidjson/document.h"
#include "rdma/rdma_common.h"
#include <etcd/Client.hpp>
#include <string>
#include <unordered_map>

class IndexHttpClient {
  public:
    IndexHttpClient(const std::string &host, int port)
        : host(host), port(port), client(host.c_str(), port) {}

    std::tuple<struct cm_con_data_t, unsigned int,
               std::vector<std::vector<int>>>
    connectRdma(uint64_t addr, uint32_t rkey, uint32_t qp_num, uint16_t lid);

  private:
    httplib::Client client;
    std::string host;
    int port;
};

extern std::unique_ptr<etcd::Client> etcdClient_;