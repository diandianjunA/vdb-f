cmake_minimum_required(VERSION 3.5) #设置cmake的最低版本
project(vdb LANGUAGES CXX) #设置项目名称 vdb
 
# 设置C++标准  基本不用改
set(CMAKE_CXX_STANDARD 20)  
# set(CMAKE_CXX_STANDARD_REQUIRED True)  

set (CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl") 

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_subdirectory(src/cuda)
set (EXTRA_LIBS ${EXTRA_LIBS} gpu)

include_directories(src/include)
include_directories(src/cuda)
link_directories(/usr/lib)
link_directories(/usr/local/lib)
link_directories(/usr/lib/x86_64-linux-gnu)
link_directories(/usr/local/cuda/lib64)

# file自定义搜索源文件，塞给集合SOURCES 
file(GLOB INDEX_SERVER_SOURCES 
    src/index_server/*.cpp 
    src/log/*.cpp 
    src/rdma/rdma_common.cpp 
    src/rdma/rdma_pools.cpp 
    src/config/*.cpp
    src/lru/*.cpp
)
file(GLOB STORAGE_SERVER_SOURCES 
    src/storage_server/*.cpp 
    src/log/*.cpp 
    src/rdma/rdma_common.cpp 
    src/rdma/rdma_pool.cpp
    src/config/*.cpp
)

add_executable(index_server src/index_server.cpp ${INDEX_SERVER_SOURCES})
add_executable(storage_server src/storage_server.cpp ${STORAGE_SERVER_SOURCES})

target_link_libraries(index_server PRIVATE ${EXTRA_LIBS}
    ibverbs etcd-cpp-api cpprest spdlog cuda cudart
)
target_link_libraries(storage_server PRIVATE
    ibverbs 
    spdlog
    rocksdb snappy z bz2 zstd lz4
)