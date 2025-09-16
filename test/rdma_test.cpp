#include <gtest/gtest.h>
#include "log/logger.h"
#include "rdma/rdma_common.h"
#include "rdma/rdma_pools.h"
#include <iostream>
#include <cuda_runtime.h>

TEST(RdmaTest, RdmaDeviceInit) {
    device = std::make_shared<RdmaDevice>();
    EXPECT_EQ(device->get_ib_port(), 1);
    EXPECT_EQ(device->get_lid(), 54);
}

TEST(RdmaTest, RdmaBufferInit) {
    size_t buf_size = 1024;
    void* buf = (void*) malloc(buf_size);
    auto rdma_buffer = std::make_shared<RdmaBuffer>(buf, buf_size);
    EXPECT_EQ(rdma_buffer->get_buf(), buf);
    EXPECT_EQ(rdma_buffer->get_size(), buf_size);

    free(buf);
}

TEST(RdmaTest, GPURdmaBufferInit) {
    size_t buf_size = 1024;
    void *buf;
    cudaMalloc(&buf, buf_size);
    auto rdma_buffer = std::make_shared<RdmaBuffer>(buf, buf_size);
    EXPECT_EQ(rdma_buffer->get_buf(), buf);
    EXPECT_EQ(rdma_buffer->get_size(), buf_size);

    cudaFree(buf);
}

TEST(RdmaTest, QPEntryInit) {
    size_t buf_size = 1024;
    void* buf = (void*) malloc(buf_size);
    auto rdma_buffer = std::make_shared<RdmaBuffer>(buf, buf_size);

    auto qp_entry = std::make_shared<QPEntry>(rdma_buffer->get_pd());
}

TEST(RdmaTest, QPEntryConnect) {
    size_t buf_size = 1024;
    void* buf1 = (void*) malloc(buf_size);
    auto rdma_buffer1 = std::make_shared<RdmaBuffer>(buf1, buf_size);
    auto qp_entry = std::make_shared<QPEntry>(rdma_buffer1->get_pd());
    
    void* buf2 = (void*) malloc(buf_size);
    auto rdma_buffer2 = std::make_shared<RdmaBuffer>(buf2, buf_size);
    auto qp_entry2 = std::make_shared<QPEntry>(rdma_buffer2->get_pd());

    struct cm_con_data_t local_props;
    struct cm_con_data_t remote_props;
    local_props.addr = (uint64_t) rdma_buffer1->get_buf();
    local_props.rkey = rdma_buffer1->get_mr()->rkey;
    local_props.qp_num = qp_entry->get_qp()->qp_num;
    local_props.lid = device->get_lid();
    remote_props.addr = (uint64_t) rdma_buffer2->get_buf();
    remote_props.rkey = rdma_buffer2->get_mr()->rkey;
    remote_props.qp_num = qp_entry2->get_qp()->qp_num;
    remote_props.lid = device->get_lid();
    
    std::thread t1([&]() {
        qp_entry->connect_qp(remote_props);
    });
    std::thread t2([&]() {
        qp_entry2->connect_qp(local_props);
    });
    t1.join();
    t2.join();
}

TEST(RdmaTest, QPEntryRead) {
    size_t buf_size = 1024;
    void* buf1 = (void*) malloc(buf_size);
    auto rdma_buffer1 = std::make_shared<RdmaBuffer>(buf1, buf_size);
    auto qp_entry = std::make_shared<QPEntry>(rdma_buffer1->get_pd());
    
    void* buf2 = (void*) malloc(buf_size);
    auto rdma_buffer2 = std::make_shared<RdmaBuffer>(buf2, buf_size);
    auto qp_entry2 = std::make_shared<QPEntry>(rdma_buffer2->get_pd());

    struct cm_con_data_t local_props;
    struct cm_con_data_t remote_props;
    local_props.addr = (uint64_t) rdma_buffer1->get_buf();
    local_props.rkey = rdma_buffer1->get_mr()->rkey;
    local_props.qp_num = qp_entry->get_qp()->qp_num;
    local_props.lid = device->get_lid();
    remote_props.addr = (uint64_t) rdma_buffer2->get_buf();
    remote_props.rkey = rdma_buffer2->get_mr()->rkey;
    remote_props.qp_num = qp_entry2->get_qp()->qp_num;
    remote_props.lid = device->get_lid();
    
    std::thread t1([&]() {
        qp_entry->connect_qp(remote_props);
        struct ibv_sge list = {
            .addr = (uint64_t) rdma_buffer1->get_buf(),
            .length = buf_size,
            .lkey = rdma_buffer1->get_mr()->lkey,
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
                            .remote_addr = qp_entry->get_remote_props()->addr + 0,
                            .rkey = qp_entry->get_remote_props()->rkey,
                        },
                },
        };
        struct ibv_send_wr *bad_wr;
        int rc;
        if (rc = ibv_post_send(qp_entry->get_qp(), &wr, &bad_wr)) {
            GlobalLogger->error("failed to post read WR, rc = {}", rc);
            throw std::runtime_error("failed to post read WR");
        }
        rc = qp_entry->poll_completion();
        if (rc) {
            GlobalLogger->error("failed to poll completion, rc = {}", rc);
            throw std::runtime_error("failed to poll completion");
        }

        EXPECT_STREQ((char*) rdma_buffer1->get_buf(), "Hello, RDMA!");
    });
    std::thread t2([&]() {
        char *buf = (char *) rdma_buffer2->get_buf();
        strcpy(buf, "Hello, RDMA!");
        EXPECT_STREQ(buf, "Hello, RDMA!");

        qp_entry2->connect_qp(local_props);
    });
    t1.join();
    t2.join();
}

TEST(RdmaTest, RdmaPoolTest) {
    size_t buf_size = 1024;
    void* buf = (void*) malloc(buf_size);
    auto rdma_pool = std::make_shared<RdmaPools>(1, buf, buf_size);
    
    rdma_pool->add_qp();
    EXPECT_EQ(rdma_pool->get_size(), 2);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    init_global_logger();
    set_log_level(spdlog::level::debug);
    return RUN_ALL_TESTS();
}