cmake_minimum_required(VERSION 3.28) #设置cmake的最低版本
project(vdb LANGUAGES CXX) #设置项目名称 vdb
 
# 设置C++标准  基本不用改
set(CMAKE_CXX_STANDARD 20)  
# set(CMAKE_CXX_STANDARD_REQUIRED True)  

set (CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl") 

include(GoogleTest)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

include_directories(src/cuda)
link_directories(./lib)
link_directories(/usr/local/lib)
link_directories(/usr/lib/x86_64-linux-gnu)
link_directories(/usr/local/cuda/lib64)

# file自定义搜索源文件，塞给集合SOURCES 
file(GLOB INIT_TEST_SOURCES 
    ../src/log/*.cpp 
    ../src/config/*.cpp
)

file(GLOB LRU_TEST_SOURCES 
    ../src/log/*.cpp
    ../src/lru/*.cpp
    ../src/config/*.cpp
)

file(GLOB RDMA_TEST_SOURCES 
    ../src/log/*.cpp 
    ../src/rdma/rdma_common.cpp
    ../src/rdma/rdma_pools.cpp
    ../src/index_server/index_http_client.cpp
)

file(GLOB SERVER_TEST_SOURCES
    ../src/log/*.cpp
)

add_executable(init_test init_test.cpp ${INIT_TEST_SOURCES})

add_executable(cuda_test cuda_test.cpp)

add_executable(lru_test lru_test.cpp ${LRU_TEST_SOURCES})

add_executable(rdma_test rdma_test.cpp ${RDMA_TEST_SOURCES})

add_executable(hnsw_test hnsw_test.cpp)

add_executable(server_test server_test.cpp ${SERVER_TEST_SOURCES})

target_link_libraries(init_test PRIVATE
    spdlog
    gtest
    gtest_main
    pthread
)

target_link_libraries(cuda_test PRIVATE
    gpu cuda cudart
    gtest
    gtest_main
    pthread
)

target_link_libraries(lru_test PRIVATE
    gtest
    gtest_main
    pthread
)

target_link_libraries(rdma_test PRIVATE
    ibverbs
    gtest
    gtest_main
    pthread
    cuda cudart
    cpprest
    etcd-cpp-api
)

target_link_libraries(hnsw_test PRIVATE
    gtest
    gtest_main
    pthread
)

target_link_libraries(server_test PRIVATE
    gtest
    gtest_main
    pthread
    curl
)
