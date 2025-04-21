#include "rdma/rdma_pools.h"
#include "index_server/index_http_client.h"
#include "log/logger.h"
#include "config/config.h"

std::shared_ptr<RdmaPools> pool;

RdmaPools::RdmaPools(int size, void *buf, size_t memory_region_size)
    : buffer(buf, memory_region_size), size(size) {
    for (int i = 0; i < size; i++) {
        add_qp();
    }
    GlobalLogger->info("QP pool initialized");
}

int RdmaPools::rdma_read(int qp_id, char *local_addr, size_t size,
                         size_t offset) {
    if (qp_id >= size) {
        GlobalLogger->error("QP id out of range");
        return -1;
    }
    QPEntry *entry = &qps[qp_id];
    std::lock_guard<std::mutex> lock(entry->get_mtx());
    if (!entry->is_connected()) {
        connect_qp(qp_id);
    }
    // GlobalLogger->debug("RDMA read: local_addr = {:#x}, size = {}, offset = {}",
    //                     (uint64_t)local_addr, size, offset);
    struct ibv_sge list = {
        .addr = (uint64_t)local_addr,
        .length = size,
        .lkey = buffer.get_mr()->lkey,
    };
    struct ibv_send_wr wr = {
        .wr_id = 0,
        .sg_list = &list,
        .num_sge = 1,
        .opcode = IBV_WR_RDMA_READ,
        .send_flags = IBV_SEND_SIGNALED,
        .wr =
            {
                .rdma =
                    {
                        .remote_addr = entry->get_remote_props()->addr + offset,
                        .rkey = entry->get_remote_props()->rkey,
                    },
            },
    };
    struct ibv_send_wr *bad_wr;
    rdma_request++;
    std::chrono::high_resolution_clock::time_point start_time =
        std::chrono::high_resolution_clock::now();
    int rc;
    if (rc = ibv_post_send(entry->get_qp(), &wr, &bad_wr)) {
        GlobalLogger->error("failed to post read WR, rc = {}", rc);
        return -1;
    }
    rc = entry->poll_completion();
    if (rc == -1) {
        GlobalLogger->error("failed to poll CQ");
        GlobalLogger->error("local_addr = {:#x}, size = {}, offset = {}",
                            (uint64_t)local_addr, size, offset);
        GlobalLogger->error("remote_addr = {:#x}, rkey = {:#x}",
                            entry->get_remote_props()->addr + offset,
                            entry->get_remote_props()->rkey);
        GlobalLogger->error("qp_num = {:#x}, lid = {:#x}",
                            entry->get_qp()->qp_num, device->get_lid());
        GlobalLogger->error("rdma_request = {}", rdma_request);
        return -1;
    }
    std::chrono::high_resolution_clock::time_point end_time =
        std::chrono::high_resolution_clock::now();
    rdma_elpased_time +=
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end_time - start_time);
    return 0;
}

QPEntry *RdmaPools::add_qp() {
    qps.emplace_back(buffer.get_pd());
    return &qps.back();
}

int RdmaPools::get_entry_node(int qp_id) {
    if (qp_id >= size) {
        GlobalLogger->error("QP id out of range");
        return -1;
    }
    QPEntry *entry = &qps[qp_id];
    std::lock_guard<std::mutex> lock(entry->get_mtx());
    if (!entry->is_connected()) {
        connect_qp(qp_id);
    }
    return entry->get_entry_node();
}

const std::vector<std::vector<int>>& RdmaPools::get_neighbors(int qp_id) {
    QPEntry *entry = &qps[qp_id];
    std::lock_guard<std::mutex> lock(entry->get_mtx());
    if (!entry->is_connected()) {
        connect_qp(qp_id);
    }
    return entry->get_neighbors();
}


void RdmaPools::connect_qp(int qp_id) {
    QPEntry *entry = &qps[qp_id];
    uint64_t addr = (uint64_t)buffer.get_buf();
    uint32_t rkey = buffer.get_mr()->rkey;
    uint32_t qp_num = entry->get_qp()->qp_num;
    uint16_t lid = device->get_lid();
    // std::string etcdKey = "/shared/" + qp_id;
    // etcd::Response etcdResponse = etcdClient_->get(etcdKey).get();
    // if (!etcdResponse.is_ok()) {
    //     GlobalLogger->error("Failed to get RDMA connection info from etcd");
    //     return -1;
    // }
    // // 解析节点信息
    // rapidjson::Document nodeDoc;
    // nodeDoc.Parse(etcdResponse.value().as_string().c_str());
    // if (!nodeDoc.IsObject()) {
    //     GlobalLogger->error("Invalid JSON response");
    //     return -1;
    // }
    // std::string host = nodeDoc["host"].GetString();
    // int port = nodeDoc["port"].GetInt();
    std::string host = "192.168.6.201";
    int port = 3000;
    auto [remote_props, entry_node, neighbor] =
        IndexHttpClient(host, port).connectRdma(addr, rkey, qp_num, lid);
    entry->set_entry_node(entry_node);
    entry->set_neighbors(std::move(neighbor));
    entry->connect_qp(remote_props);
}