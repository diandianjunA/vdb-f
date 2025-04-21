#include "storage_server/neighbor_base.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

void Neighbor::update_neighbors(hnswlib::HierarchicalNSW<float> *index) {
    for (int i = 0; i < num_data_; i++) {
        int *ptr = (int *)index->get_linklist0(i);
        int size = index->getListCount((unsigned int *)ptr);
        std::vector<int> neighbors(size);
        for (int j = 1; j <= size; j++) {
            neighbors[j - 1] = ptr[j];
        }
        neighbors_[i] = neighbors;
    }
}

std::string Neighbor::serialize() {
    rapidjson::Document doc(rapidjson::kArrayType);
    rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

    for (const auto &inner_vec : neighbors_) {
        rapidjson::Value inner_array(rapidjson::kArrayType);
        for (int num : inner_vec) {
            inner_array.PushBack(num, allocator);
        }
        doc.PushBack(inner_array, allocator);
    }

    rapidjson::StringBuffer buffer;
    rapidjson:: Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    return buffer.GetString();
}