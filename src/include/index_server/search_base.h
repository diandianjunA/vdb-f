#pragma once
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>

struct Node {

    Node() {
        distance = 0;
        nodeid = 0;
    }

    Node(int id, float dist) {
        distance = dist;
        nodeid = id;
    }

    Node(const Node &other) {
        distance = other.distance;
        nodeid = other.nodeid;
    }

    Node(Node &&other) {
        distance = other.distance;
        nodeid = other.nodeid;
    }
    Node &operator=(const Node &other) {
        if (this != &other) {
            distance = other.distance;
            nodeid = other.nodeid;
        }
        return *this;
    }
    Node &operator=(Node &&other) {
        if (this != &other) {
            distance = other.distance;
            nodeid = other.nodeid;
        }
        return *this;
    }

    float distance;
    int nodeid;

    bool operator<(const Node &other) const {
        return distance < other.distance;
    }

    bool operator==(const Node &other) const {
        return (nodeid == other.nodeid) || (distance == other.distance);
    }

    bool operator>(const Node &other) const {
        return distance > other.distance;
    }

    bool operator>=(const Node &other) const {
        return distance >= other.distance;
    }

    bool operator<=(const Node &other) const {
        return distance <= other.distance;
    }

    bool operator!=(const Node &other) const {
        return distance != other.distance;
    }
};

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

float calculate_distance(const float *a, const float *b, int dim);

void search(unsigned int entry_node, const float *query, int k, int ef_search, std::vector<long> &ids,
       std::vector<float> &, int shard);