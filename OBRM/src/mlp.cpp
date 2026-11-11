#include "mlp.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace {

float relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}

float relu_grad(float x)
{
    return x > 0.0f ? 1.0f : 0.0f;
}

} // namespace

void MLP::train(const std::vector<float>& features, const std::vector<int32_t>& labels,
    std::size_t sample_count, std::size_t input_dim, std::size_t output_dim,
    std::size_t hidden_dim, std::size_t epochs, float learning_rate, std::mt19937& rng)
{
    if (sample_count == 0 || input_dim == 0 || output_dim == 0) {
        throw std::runtime_error("invalid MLP training shape");
    }
    if (features.size() != sample_count * input_dim || labels.size() != sample_count) {
        throw std::runtime_error("MLP training data shape mismatch");
    }

    input_dim_ = input_dim;
    hidden_dim_ = hidden_dim;
    output_dim_ = output_dim;
    mean_.assign(input_dim_, 0.0f);
    inv_std_.assign(input_dim_, 1.0f);

    for (std::size_t i = 0; i < sample_count; ++i) {
        for (std::size_t d = 0; d < input_dim_; ++d) {
            mean_[d] += features[i * input_dim_ + d];
        }
    }
    for (float& v : mean_) {
        v /= static_cast<float>(sample_count);
    }
    std::vector<float> var(input_dim_, 0.0f);
    for (std::size_t i = 0; i < sample_count; ++i) {
        for (std::size_t d = 0; d < input_dim_; ++d) {
            const float diff = features[i * input_dim_ + d] - mean_[d];
            var[d] += diff * diff;
        }
    }
    for (std::size_t d = 0; d < input_dim_; ++d) {
        const float stddev = std::sqrt(var[d] / static_cast<float>(sample_count) + 1e-6f);
        inv_std_[d] = 1.0f / stddev;
    }

    std::normal_distribution<float> init(0.0f, 0.05f);
    w1_.resize(input_dim_ * hidden_dim_);
    b1_.assign(hidden_dim_, 0.0f);
    w2_.resize(hidden_dim_ * output_dim_);
    b2_.assign(output_dim_, 0.0f);
    for (float& v : w1_) {
        v = init(rng);
    }
    for (float& v : w2_) {
        v = init(rng);
    }

    std::vector<std::size_t> order(sample_count);
    std::iota(order.begin(), order.end(), 0);
    std::vector<float> x(input_dim_);
    std::vector<float> hidden(hidden_dim_);
    std::vector<float> pre_hidden(hidden_dim_);
    std::vector<float> logits(output_dim_);
    std::vector<float> probs(output_dim_);
    std::vector<float> grad_hidden(hidden_dim_);

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        const float lr = learning_rate / (1.0f + 0.05f * static_cast<float>(epoch));
        for (const std::size_t idx : order) {
            for (std::size_t d = 0; d < input_dim_; ++d) {
                x[d] = (features[idx * input_dim_ + d] - mean_[d]) * inv_std_[d];
            }

            for (std::size_t h = 0; h < hidden_dim_; ++h) {
                float z = b1_[h];
                for (std::size_t d = 0; d < input_dim_; ++d) {
                    z += x[d] * w1_[d * hidden_dim_ + h];
                }
                pre_hidden[h] = z;
                hidden[h] = relu(z);
            }

            float max_logit = -1e30f;
            for (std::size_t o = 0; o < output_dim_; ++o) {
                float z = b2_[o];
                for (std::size_t h = 0; h < hidden_dim_; ++h) {
                    z += hidden[h] * w2_[h * output_dim_ + o];
                }
                logits[o] = z;
                max_logit = std::max(max_logit, z);
            }
            float denom = 0.0f;
            for (std::size_t o = 0; o < output_dim_; ++o) {
                probs[o] = std::exp(logits[o] - max_logit);
                denom += probs[o];
            }
            for (float& p : probs) {
                p /= denom;
            }

            const auto label = static_cast<std::size_t>(labels[idx]);
            if (label >= output_dim_) {
                throw std::runtime_error("MLP label out of range");
            }
            probs[label] -= 1.0f;

            std::fill(grad_hidden.begin(), grad_hidden.end(), 0.0f);
            for (std::size_t h = 0; h < hidden_dim_; ++h) {
                for (std::size_t o = 0; o < output_dim_; ++o) {
                    grad_hidden[h] += probs[o] * w2_[h * output_dim_ + o];
                    w2_[h * output_dim_ + o] -= lr * hidden[h] * probs[o];
                }
            }
            for (std::size_t o = 0; o < output_dim_; ++o) {
                b2_[o] -= lr * probs[o];
            }

            for (std::size_t h = 0; h < hidden_dim_; ++h) {
                const float gh = grad_hidden[h] * relu_grad(pre_hidden[h]);
                for (std::size_t d = 0; d < input_dim_; ++d) {
                    w1_[d * hidden_dim_ + h] -= lr * x[d] * gh;
                }
                b1_[h] -= lr * gh;
            }
        }
    }
}

void MLP::normalize_in_place(std::vector<float>& feature) const
{
    if (feature.size() != input_dim_) {
        throw std::runtime_error("MLP feature dimension mismatch");
    }
    for (std::size_t d = 0; d < input_dim_; ++d) {
        feature[d] = (feature[d] - mean_[d]) * inv_std_[d];
    }
}

std::vector<float> MLP::predict(const std::vector<float>& feature) const
{
    if (input_dim_ == 0) {
        throw std::runtime_error("untrained MLP");
    }
    std::vector<float> x = feature;
    normalize_in_place(x);

    std::vector<float> hidden(hidden_dim_);
    for (std::size_t h = 0; h < hidden_dim_; ++h) {
        float z = b1_[h];
        for (std::size_t d = 0; d < input_dim_; ++d) {
            z += x[d] * w1_[d * hidden_dim_ + h];
        }
        hidden[h] = relu(z);
    }

    std::vector<float> probs(output_dim_);
    float max_logit = -1e30f;
    for (std::size_t o = 0; o < output_dim_; ++o) {
        float z = b2_[o];
        for (std::size_t h = 0; h < hidden_dim_; ++h) {
            z += hidden[h] * w2_[h * output_dim_ + o];
        }
        probs[o] = z;
        max_logit = std::max(max_logit, z);
    }
    float denom = 0.0f;
    for (float& p : probs) {
        p = std::exp(p - max_logit);
        denom += p;
    }
    for (float& p : probs) {
        p /= denom;
    }
    return probs;
}

std::vector<int32_t> MLP::top_k(const std::vector<float>& feature, std::size_t k) const
{
    auto probs = predict(feature);
    std::vector<int32_t> ids(probs.size());
    std::iota(ids.begin(), ids.end(), 0);
    const std::size_t keep = std::min(k, ids.size());
    std::partial_sort(ids.begin(), ids.begin() + keep, ids.end(),
        [&](int32_t a, int32_t b) { return probs[static_cast<std::size_t>(a)] > probs[static_cast<std::size_t>(b)]; });
    ids.resize(keep);
    return ids;
}

std::size_t MLP::parameter_count() const
{
    return w1_.size() + b1_.size() + w2_.size() + b2_.size() + mean_.size() + inv_std_.size();
}

std::size_t MLP::serialized_bytes() const
{
    return sizeof(uint64_t) * 3
        + sizeof(float) * (w1_.size() + b1_.size() + w2_.size() + b2_.size() + mean_.size() + inv_std_.size());
}

void MLP::append_parameters(std::vector<float>& out) const
{
    out.insert(out.end(), w1_.begin(), w1_.end());
    out.insert(out.end(), b1_.begin(), b1_.end());
    out.insert(out.end(), w2_.begin(), w2_.end());
    out.insert(out.end(), b2_.begin(), b2_.end());
    out.insert(out.end(), mean_.begin(), mean_.end());
    out.insert(out.end(), inv_std_.begin(), inv_std_.end());
}
