#include "learned_ann.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_set>

namespace {

std::size_t non_empty_class_count(const std::vector<int32_t>& labels, std::size_t K)
{
    std::vector<bool> seen(K, false);
    for (int32_t label : labels) {
        if (label >= 0 && static_cast<std::size_t>(label) < K) {
            seen[static_cast<std::size_t>(label)] = true;
        }
    }
    return static_cast<std::size_t>(std::count(seen.begin(), seen.end(), true));
}

} // namespace

void LearnedAnnIndex::build(const FloatVectors& base, const LearnedAnnParams& params)
{
    const auto build_start = std::chrono::steady_clock::now();
    if (base.count == 0 || base.dim == 0) {
        throw std::runtime_error("empty base vectors");
    }
    if (params.K == 0 || params.leaf_threshold == 0 || params.block_size == 0 || params.pivot_count == 0) {
        throw std::runtime_error("invalid learned ANN parameters");
    }
    base_ = &base;
    params_ = params;
    if (params_.internal_pivot_count == 0) {
        params_.internal_pivot_count = params_.pivot_count;
    }
    if (params_.leaf_pivot_count == 0) {
        params_.leaf_pivot_count = params_.pivot_count;
    }
    nodes_.clear();
    stats_ = {};
    rng_ = std::mt19937(params.seed);

    std::vector<int32_t> ids(base.count);
    std::iota(ids.begin(), ids.end(), 0);
    build_node(ids, 0);
    const auto build_end = std::chrono::steady_clock::now();
    const double total_seconds = std::chrono::duration<double>(build_end - build_start).count();
    stats_.index_construction_seconds = std::max(0.0, total_seconds - stats_.model_training_seconds);
}

int32_t LearnedAnnIndex::build_node(const std::vector<int32_t>& point_ids, std::size_t depth)
{
    const int32_t node_id = static_cast<int32_t>(nodes_.size());
    nodes_.emplace_back();
    stats_.max_depth = std::max(stats_.max_depth, depth);
    Node& node = nodes_.back();

    if (point_ids.size() <= params_.leaf_threshold || point_ids.size() <= params_.K) {
        node.pivot_ids = select_pivots(point_ids, std::min(params_.leaf_pivot_count, point_ids.size()));
        node.base_feature = pivot_feature_for_point(point_ids.front(), node.pivot_ids);
        node.is_leaf = true;
        ++stats_.leaf_nodes;
        build_leaf(node, point_ids);
        return node_id;
    }

    node.pivot_ids = select_pivots(point_ids, std::min(params_.internal_pivot_count, point_ids.size()));
    node.base_feature = pivot_feature_for_point(point_ids.front(), node.pivot_ids);
    const auto labels = kmeans(point_ids, params_.K);
    const std::size_t classes = non_empty_class_count(labels, params_.K);
    if (classes <= 1) {
        node.pivot_ids = select_pivots(point_ids, std::min(params_.leaf_pivot_count, point_ids.size()));
        node.base_feature = pivot_feature_for_point(point_ids.front(), node.pivot_ids);
        node.is_leaf = true;
        ++stats_.leaf_nodes;
        build_leaf(node, point_ids);
        return node_id;
    }

    node.is_leaf = false;
    ++stats_.internal_nodes;
    node.children.assign(params_.K, -1);
    train_node_model(node, point_ids, labels, params_.K);

    std::vector<std::vector<int32_t>> buckets(params_.K);
    for (std::size_t i = 0; i < point_ids.size(); ++i) {
        buckets[static_cast<std::size_t>(labels[i])].push_back(point_ids[i]);
    }
    for (std::size_t c = 0; c < params_.K; ++c) {
        if (!buckets[c].empty()) {
            const int32_t child_id = build_node(buckets[c], depth + 1);
            nodes_[static_cast<std::size_t>(node_id)].children[c] = child_id;
        }
    }
    return node_id;
}

std::vector<int32_t> LearnedAnnIndex::select_pivots(const std::vector<int32_t>& point_ids, std::size_t count)
{
    if (point_ids.empty()) {
        return {};
    }
    std::vector<int32_t> pivots;
    pivots.reserve(count);
    std::uniform_int_distribution<std::size_t> pick(0, point_ids.size() - 1);
    pivots.push_back(point_ids[pick(rng_)]);

    std::vector<float> nearest(point_ids.size(), std::numeric_limits<float>::max());
    for (std::size_t p = 1; p < count; ++p) {
        const int32_t last = pivots.back();
        const float* last_vec = base_->values.data() + static_cast<std::size_t>(last) * base_->dim;
        std::size_t best_i = 0;
        float best_dist = -1.0f;
        for (std::size_t i = 0; i < point_ids.size(); ++i) {
            const int32_t id = point_ids[i];
            const float* candidate = base_->values.data() + static_cast<std::size_t>(id) * base_->dim;
            const float dist = point_center_distance(candidate, last_vec);
            nearest[i] = std::min(nearest[i], dist);
            if (nearest[i] > best_dist) {
                best_dist = nearest[i];
                best_i = i;
            }
        }
        pivots.push_back(point_ids[best_i]);
    }
    return pivots;
}

std::vector<float> LearnedAnnIndex::pivot_feature(const float* vector, const std::vector<int32_t>& pivot_ids) const
{
    std::vector<float> feature(pivot_ids.size());
    for (std::size_t i = 0; i < pivot_ids.size(); ++i) {
        const int32_t pivot = pivot_ids[i];
        feature[i] = std::sqrt(squared_l2(vector, base_->values.data() + static_cast<std::size_t>(pivot) * base_->dim, base_->dim));
    }
    return feature;
}

std::vector<float> LearnedAnnIndex::pivot_feature_for_point(int32_t point_id, const std::vector<int32_t>& pivot_ids) const
{
    return pivot_feature(base_->values.data() + static_cast<std::size_t>(point_id) * base_->dim, pivot_ids);
}

float LearnedAnnIndex::point_center_distance(const float* point, const float* center) const
{
    return squared_l2(point, center, base_->dim);
}

std::vector<int32_t> LearnedAnnIndex::normal_kmeans(const std::vector<int32_t>& point_ids, std::size_t K)
{
    K = std::min(K, point_ids.size());
    const auto centers_ids = select_pivots(point_ids, K);
    std::vector<float> centers(K * base_->dim);
    for (std::size_t c = 0; c < K; ++c) {
        const float* src = base_->values.data() + static_cast<std::size_t>(centers_ids[c]) * base_->dim;
        std::copy(src, src + base_->dim, centers.data() + c * base_->dim);
    }

    std::vector<int32_t> labels(point_ids.size(), 0);
    std::vector<float> sums(K * base_->dim);
    std::vector<std::size_t> counts(K);

    for (std::size_t iter = 0; iter < params_.kmeans_iters; ++iter) {
        std::fill(sums.begin(), sums.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (std::size_t i = 0; i < point_ids.size(); ++i) {
            const float* v = base_->values.data() + static_cast<std::size_t>(point_ids[i]) * base_->dim;
            std::size_t best = 0;
            float best_dist = std::numeric_limits<float>::max();
            for (std::size_t c = 0; c < K; ++c) {
                const float dist = point_center_distance(v, centers.data() + c * base_->dim);
                if (dist < best_dist) {
                    best_dist = dist;
                    best = c;
                }
            }
            labels[i] = static_cast<int32_t>(best);
            ++counts[best];
            float* sum = sums.data() + best * base_->dim;
            for (std::size_t d = 0; d < base_->dim; ++d) {
                sum[d] += v[d];
            }
        }

        for (std::size_t c = 0; c < K; ++c) {
            if (counts[c] == 0) {
                continue;
            }
            float* center = centers.data() + c * base_->dim;
            const float* sum = sums.data() + c * base_->dim;
            const float inv = 1.0f / static_cast<float>(counts[c]);
            for (std::size_t d = 0; d < base_->dim; ++d) {
                center[d] = sum[d] * inv;
            }
        }
    }
    return labels;
}

std::vector<int32_t> LearnedAnnIndex::kmeans(const std::vector<int32_t>& point_ids, std::size_t K)
{
    return normal_kmeans(point_ids, K);
}

void LearnedAnnIndex::build_leaf(Node& node, const std::vector<int32_t>& point_ids)
{
    const std::size_t block_count = (point_ids.size() + params_.block_size - 1) / params_.block_size;
    const auto cluster_labels = kmeans(point_ids, block_count);
    std::vector<std::vector<std::size_t>> clusters(block_count);
    for (std::size_t i = 0; i < point_ids.size(); ++i) {
        clusters[static_cast<std::size_t>(cluster_labels[i])].push_back(i);
    }

    std::vector<int32_t> train_ids;
    std::vector<int32_t> labels;
    train_ids.reserve(point_ids.size());
    labels.reserve(point_ids.size());

    for (auto& cluster : clusters) {
        if (cluster.empty()) {
            continue;
        }

        std::vector<float> center(base_->dim, 0.0f);
        for (std::size_t local_i : cluster) {
            const float* v = base_->values.data() + static_cast<std::size_t>(point_ids[local_i]) * base_->dim;
            for (std::size_t d = 0; d < base_->dim; ++d) {
                center[d] += v[d];
            }
        }
        const float inv = 1.0f / static_cast<float>(cluster.size());
        for (float& value : center) {
            value *= inv;
        }

        std::sort(cluster.begin(), cluster.end(), [&](std::size_t a, std::size_t b) {
            const float* va = base_->values.data() + static_cast<std::size_t>(point_ids[a]) * base_->dim;
            const float* vb = base_->values.data() + static_cast<std::size_t>(point_ids[b]) * base_->dim;
            float da = point_center_distance(va, center.data());
            float db = point_center_distance(vb, center.data());
            return da < db;
        });

        for (std::size_t offset = 0; offset < cluster.size(); offset += params_.block_size) {
            const int32_t block_id = static_cast<int32_t>(node.blocks.size());
            DataBlock block;
            const std::size_t end = std::min(cluster.size(), offset + params_.block_size);
            for (std::size_t i = offset; i < end; ++i) {
                const int32_t point_id = point_ids[cluster[i]];
                block.point_ids.push_back(point_id);
                train_ids.push_back(point_id);
                labels.push_back(block_id);
            }
            node.blocks.push_back(std::move(block));
        }
    }


    stats_.data_blocks += node.blocks.size();
    train_node_model(node, train_ids, labels, node.blocks.size());
}

void LearnedAnnIndex::train_node_model(Node& node, const std::vector<int32_t>& point_ids,
    const std::vector<int32_t>& labels, std::size_t output_dim)
{
    std::vector<float> features(point_ids.size() * node.pivot_ids.size());
    for (std::size_t i = 0; i < point_ids.size(); ++i) {
        const auto f = pivot_feature_for_point(point_ids[i], node.pivot_ids);
        std::copy(f.begin(), f.end(), features.begin() + static_cast<std::ptrdiff_t>(i * node.pivot_ids.size()));
    }
    const auto training_start = std::chrono::steady_clock::now();
    node.model.train(features, labels, point_ids.size(), node.pivot_ids.size(), output_dim,
        params_.hidden_dim, params_.epochs, params_.learning_rate, rng_);
    const auto training_end = std::chrono::steady_clock::now();
    stats_.model_training_seconds += std::chrono::duration<double>(training_end - training_start).count();
    stats_.model_parameters += node.model.parameter_count();
}

LearnedAnnSnapshot LearnedAnnIndex::snapshot() const
{
    LearnedAnnSnapshot out;
    out.nodes.reserve(nodes_.size());
    std::vector<std::vector<int32_t>> leaf_block_ids(nodes_.size());
    int32_t next_data_id = static_cast<int32_t>(nodes_.size());

    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        if (!nodes_[i].is_leaf) {
            continue;
        }
        leaf_block_ids[i].reserve(nodes_[i].blocks.size());
        for (const auto& block : nodes_[i].blocks) {
            LearnedDataBlock data;
            data.logical_id = next_data_id++;
            data.point_ids = block.point_ids;
            leaf_block_ids[i].push_back(data.logical_id);
            out.data_blocks.push_back(std::move(data));
        }
    }

    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        const auto& node = nodes_[i];
        LearnedNodeBlock block;
        block.logical_id = static_cast<int32_t>(i);
        block.is_leaf = node.is_leaf;
        block.pivot_ids = node.pivot_ids;
        block.model = node.model;
        if (node.is_leaf) {
            block.data_block_ids = leaf_block_ids[i];
        } else {
            block.child_node_ids = node.children;
        }
        out.nodes.push_back(std::move(block));
    }
    return out;
}
