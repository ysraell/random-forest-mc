#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <cmath>

namespace rf_mc {

enum class FeatType : uint8_t {
    NUMERIC = 0,
    CATEGORICAL = 1
};

struct SplitResult {
    bool valid = false;
    double split_val = 0.0;
    std::vector<int32_t> idx_ge; // >= or ==
    std::vector<int32_t> idx_lt; // < or !=
};

} // namespace rf_mc
