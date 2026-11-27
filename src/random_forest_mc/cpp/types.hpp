#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <cmath>

namespace rf_mc {

enum class FeatType : uint8_t {
    NUMERIC = 0,
    CATEGORICAL = 1,
    CATEGORICAL_SET = 2
};

struct SplitResult {
    bool valid = false;
    FeatType feat_type = FeatType::NUMERIC;
    double split_val = 0.0;
    std::vector<double> split_set;  // CATEGORICAL_SET: codes routed to '>='
    std::vector<int32_t> idx_ge;    // >=, == or in set
    std::vector<int32_t> idx_lt;    // <, != or not in set
};

// Tree-growing hyperparameters (mirrors the Python engine).
struct TreeParams {
    int32_t max_depth = 1000;
    int32_t min_samples_split = 1;
    int32_t n_classes = 0;
    int32_t split_rule = 1;          // 0: legacy (v1.5.0), 1: midpoint
    int32_t split_supervision = 0;   // 0, 1 or 2
    int32_t split_lookahead = 2;
    int32_t categorical_split = 0;   // 0: mode-vs-rest, 1: ordered (set split)
};

// Inference options.
struct InferParams {
    int32_t missing_strategy = 0;    // 0: weighted, 1: uniform over reached leaves (v1.5.0)
    double leaf_alpha = 0.0;         // Laplace smoothing of leaf distributions
};

} // namespace rf_mc
