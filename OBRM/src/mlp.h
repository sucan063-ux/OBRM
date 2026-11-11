#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

class MLP {
public:
    void train(const std::vector<float>& features, const std::vector<int32_t>& labels,
        std::size_t sample_count, std::size_t input_dim, std::size_t output_dim,
        std::size_t hidden_dim, std::size_t epochs, float learning_rate, std::mt19937& rng);

    std::vector<float> predict(const std::vector<float>& feature) const;
    std::vector<int32_t> top_k(const std::vector<float>& feature, std::size_t k) const;

    std::size_t input_dim() const { return input_dim_; }
    std::size_t output_dim() const { return output_dim_; }
    std::size_t hidden_dim() const { return hidden_dim_; }
    std::size_t parameter_count() const;
    std::size_t serialized_bytes() const;
    void append_parameters(std::vector<float>& out) const;

private:
    std::size_t input_dim_ = 0;
    std::size_t hidden_dim_ = 0;
    std::size_t output_dim_ = 0;
    std::vector<float> mean_;
    std::vector<float> inv_std_;
    std::vector<float> w1_;
    std::vector<float> b1_;
    std::vector<float> w2_;
    std::vector<float> b2_;

    void normalize_in_place(std::vector<float>& feature) const;
};
