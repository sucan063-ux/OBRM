#include "vectors.h"
#include "VectorFormat.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {

int32_t read_dim(std::ifstream& in)
{
    int32_t dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!in) {
        throw std::runtime_error("failed to read vector dimension");
    }
    if (dim <= 0) {
        throw std::runtime_error("invalid vector dimension");
    }
    return dim;
}

std::size_t file_size(std::ifstream& in)
{
    const auto current = in.tellg();
    in.seekg(0, std::ios::end);
    const auto end = in.tellg();
    in.seekg(current, std::ios::beg);
    return static_cast<std::size_t>(end);
}

} // namespace

FloatVectors read_vectors(const std::string& path, uint32_t encoding, std::size_t limit)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open " + path);
    }

    const int32_t dim = read_dim(in);
    const size_t value_bytes=vector_format::element_bytes(encoding);
    if (!value_bytes) throw std::runtime_error("unsupported vector coordinate encoding");
    const std::string extension=encoding==vector_format::UInt8?".bvecs":".hvecs";
    if (path.size()<extension.size() || path.substr(path.size()-extension.size())!=extension)
        throw std::runtime_error("vector file extension does not match coordinate encoding");
    const std::size_t record_size = sizeof(int32_t) + value_bytes * static_cast<std::size_t>(dim);
    const std::size_t bytes = file_size(in);
    if (bytes % record_size != 0) {
        throw std::runtime_error("invalid encoded vector file size: " + path);
    }
    std::size_t count = bytes / record_size;
    if (limit != 0) {
        count = std::min(count, limit);
    }

    FloatVectors out;
    out.count = count;
    out.dim = static_cast<std::size_t>(dim);
    out.values.resize(count * out.dim);

    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> coordinates(static_cast<size_t>(dim)*value_bytes);
    for (std::size_t i = 0; i < count; ++i) {
        const int32_t current_dim = read_dim(in);
        if (current_dim != dim) {
            throw std::runtime_error("mixed dimensions in " + path);
        }
        in.read(reinterpret_cast<char*>(coordinates.data()),coordinates.size());
        if (!in) {
            throw std::runtime_error("failed to read vector payload from " + path);
        }
        for (size_t d=0;d<out.dim;++d) {
            float value=vector_format::coordinate(coordinates.data(),d,encoding);
            if (!std::isfinite(value)) throw std::runtime_error("non-finite encoded coordinate");
            out.values[i*out.dim+d]=value;
        }
    }
    return out;
}

IntVectors read_ivecs(const std::string& path, std::size_t limit)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open " + path);
    }

    const int32_t dim = read_dim(in);
    const std::size_t record_size = sizeof(int32_t) + sizeof(int32_t) * static_cast<std::size_t>(dim);
    const std::size_t bytes = file_size(in);
    if (bytes % record_size != 0) {
        throw std::runtime_error("invalid ivecs file size: " + path);
    }
    std::size_t count = bytes / record_size;
    if (limit != 0) {
        count = std::min(count, limit);
    }

    IntVectors out;
    out.count = count;
    out.dim = static_cast<std::size_t>(dim);
    out.values.resize(count * out.dim);

    in.seekg(0, std::ios::beg);
    for (std::size_t i = 0; i < count; ++i) {
        const int32_t current_dim = read_dim(in);
        if (current_dim != dim) {
            throw std::runtime_error("mixed dimensions in " + path);
        }
        in.read(reinterpret_cast<char*>(out.values.data() + i * out.dim), sizeof(int32_t) * out.dim);
        if (!in) {
            throw std::runtime_error("failed to read vector payload from " + path);
        }
    }
    return out;
}

void write_ivecs(const std::string& path, const IntVectors& vectors)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open " + path);
    }
    const int32_t dim = static_cast<int32_t>(vectors.dim);
    for (std::size_t i = 0; i < vectors.count; ++i) {
        out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        out.write(reinterpret_cast<const char*>(vectors.values.data() + i * vectors.dim),
            sizeof(int32_t) * vectors.dim);
    }
}

float squared_l2(const float* a, const float* b, std::size_t dim)
{
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

double recall_at_10_gt10(const IntVectors& results, const IntVectors& groundtruth)
{
    if (results.count != groundtruth.count || results.dim < 10 || groundtruth.dim < 10) {
        throw std::runtime_error("recall@10 requires matching counts and at least 10 neighbors");
    }

    std::size_t hits = 0;
    for (std::size_t i = 0; i < results.count; ++i) {
        const int32_t* got = results.values.data() + i * results.dim;
        const int32_t* gt = groundtruth.values.data() + i * groundtruth.dim;
        for (std::size_t a = 0; a < 10; ++a) {
            for (std::size_t b = 0; b < 10; ++b) {
                if (got[a] == gt[b]) {
                    ++hits;
                    break;
                }
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(results.count * 10);
}

