#pragma once

#include "vectors.h"
#include "mlp.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>


struct LearnedAnnParams {
    std::size_t K = 8;
    std::size_t leaf_threshold = 2048;
    std::size_t block_size = 64;
    std::size_t pivot_count = 7;
    std::size_t internal_pivot_count = 0;
    std::size_t leaf_pivot_count = 0;
    std::size_t r = 3;
    std::size_t result_k = 10;
    std::size_t hidden_dim = 64;
    std::size_t epochs = 25;
    float learning_rate = 0.01f;
    std::size_t kmeans_iters = 12;
    uint32_t seed = 20260929;
};

struct LearnedAnnStats {
    std::size_t internal_nodes = 0;
    std::size_t leaf_nodes = 0;
    std::size_t data_blocks = 0;
    std::size_t max_depth = 0;
    std::size_t model_parameters = 0;
    double index_construction_seconds = 0.0;
    double model_training_seconds = 0.0;
};

struct LearnedNodeBlock {
    int32_t logical_id = -1;
    bool is_leaf = false;
    std::vector<int32_t> pivot_ids;
    std::vector<int32_t> child_node_ids;
    std::vector<int32_t> data_block_ids;
    MLP model;
};

struct LearnedDataBlock {
    int32_t logical_id = -1;
    std::vector<int32_t> point_ids;
};

struct LearnedAnnSnapshot {
    std::vector<LearnedNodeBlock> nodes;
    std::vector<LearnedDataBlock> data_blocks;
};

class LearnedAnnIndex {
public:
    void build(const FloatVectors& base, const LearnedAnnParams& params);

    const LearnedAnnStats& stats() const { return stats_; }
    LearnedAnnSnapshot snapshot() const;

private:
    struct DataBlock {
        std::vector<int32_t> point_ids;
    };

    struct Node {
        bool is_leaf = false;
        std::vector<int32_t> pivot_ids;
        std::vector<float> base_feature;
        std::vector<int32_t> children;
        std::vector<DataBlock> blocks;
        MLP model;
    };

    const FloatVectors* base_ = nullptr;
    LearnedAnnParams params_;
    std::vector<Node> nodes_;
    LearnedAnnStats stats_;
    std::mt19937 rng_;

    int32_t build_node(const std::vector<int32_t>& point_ids, std::size_t depth);
    std::vector<int32_t> select_pivots(const std::vector<int32_t>& point_ids, std::size_t count);
    std::vector<float> pivot_feature(const float* vector, const std::vector<int32_t>& pivot_ids) const;
    std::vector<float> pivot_feature_for_point(int32_t point_id, const std::vector<int32_t>& pivot_ids) const;
    float point_center_distance(const float* point, const float* center) const;
    std::vector<int32_t> normal_kmeans(const std::vector<int32_t>& point_ids, std::size_t K);
    std::vector<int32_t> kmeans(const std::vector<int32_t>& point_ids, std::size_t K);
    void build_leaf(Node& node, const std::vector<int32_t>& point_ids);
    void train_node_model(Node& node, const std::vector<int32_t>& point_ids, const std::vector<int32_t>& labels,
        std::size_t output_dim);
};
