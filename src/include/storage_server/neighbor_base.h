#pragma once
#include "hnswlib/hnswlib.h"
#include <vector>

class Neighbor {
  public:
    Neighbor() = default;
    Neighbor(int num_data) : num_data_(num_data) {
        neighbors_.resize(num_data_);
    }
    ~Neighbor() = default;

    void update_neighbors(hnswlib::HierarchicalNSW<float> *index);
    std::string serialize();

  private:
    int num_data_;
    std::vector<std::vector<int>> neighbors_;
};