#pragma once

#include "types.hpp"
#include <vector>
#include <cstdint>

namespace rf_mc {

struct Node {
    int32_t feature_idx = -1;  // -1 means leaf
    FeatType feat_type = FeatType::NUMERIC;
    double split_val = 0.0;
    std::vector<double> split_set; // CATEGORICAL_SET only
    int32_t left_child = -1;   // Index of >= branch in tree's node vector
    int32_t right_child = -1;  // Index of < branch in tree's node vector
    int32_t depth = 0;
    int32_t n_ge = 0;          // Training rows sent to '>=' (0 when unknown: format 1 trees)
    int32_t n_lt = 0;          // Training rows sent to '<'
    int32_t n_samples = 0;     // Leaf training rows (0 when unknown)
    std::vector<double> leaf_probs; // Size = n_classes (only for leaf)

    bool is_leaf() const {
        return feature_idx == -1;
    }
};

} // namespace rf_mc
