#pragma once

#include "hnswlib/hnswlib.h"
#include <vector>

class IndexEngine {
  public:
    // 构造函数
    IndexEngine(int dim);

    // 查询向量
    std::pair<std::vector<long>, std::vector<float>>
    search_vectors(const std::vector<float> &query, int k, int ef_search = 50);

  private:
    int dim;
};