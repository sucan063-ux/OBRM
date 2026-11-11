#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// A computational view: every value is an exact float32 expansion of a stored
// uint8 or float16 coordinate. It never reads unquantized float32 vector files.
struct FloatVectors {
    std::size_t count = 0;
    std::size_t dim = 0;
    std::vector<float> values;
};

struct IntVectors {
    std::size_t count = 0;
    std::size_t dim = 0;
    std::vector<int32_t> values;
};

FloatVectors read_vectors(const std::string& path, uint32_t encoding, std::size_t limit = 0);
IntVectors read_ivecs(const std::string& path, std::size_t limit = 0);
void write_ivecs(const std::string& path, const IntVectors& vectors);

float squared_l2(const float* a, const float* b, std::size_t dim);
double recall_at_10_gt10(const IntVectors& results, const IntVectors& groundtruth);

