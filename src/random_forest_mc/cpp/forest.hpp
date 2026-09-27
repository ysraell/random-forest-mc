#pragma once

#include "tree.hpp"
#include "types.hpp"
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/unordered_map.h>

namespace nb = nanobind;

namespace rf_mc {

class RandomForestCPP {
public:
    int32_t n_trees = 16;
    std::string target_col = "target";
    int32_t batch_train_pclass = 10;
    int32_t batch_val_pclass = 10;
    int32_t max_discard_trees = 10;
    double delta_th = 0.1;
    double th_start = 1.0;
    bool get_best_tree = true;
    int32_t min_feature = -1;
    int32_t max_feature = -1;
    bool temporal_features = false;
    int32_t max_depth = 1000;
    int32_t min_samples_split = 1;
    bool soft_voting = false;
    bool weighted_tree = false;
    uint64_t random_seed = 42;

    std::vector<DecisionTreeCPP> trees;
    std::vector<double> survived_scores;
    std::vector<std::string> feature_names;
    std::vector<std::string> class_names;
    std::vector<FeatType> feature_types;

    RandomForestCPP() = default;

    void set_features_and_classes(
        const std::vector<std::string>& f_names,
        const std::vector<std::string>& c_names,
        const std::vector<std::string>& f_types
    );

    void fit(
        const double* X,
        const int32_t* y,
        size_t n_rows,
        size_t n_cols,
        int32_t n_threads
    );

    void predict_row_proba(
        const double* row,
        size_t n_cols,
        std::vector<double>& out_probs
    ) const;

    int32_t predict_row_class(
        const double* row,
        size_t n_cols
    ) const;

    void predict_batch(
        const double* X,
        size_t n_rows,
        size_t n_cols,
        std::vector<int32_t>& out_classes,
        int32_t n_threads
    ) const;

    void predict_proba_batch(
        const double* X,
        size_t n_rows,
        size_t n_cols,
        std::vector<double>& out_probs_flat,
        int32_t n_threads
    ) const;

    // Feature analysis
    std::unordered_map<std::string, double> feat_importance() const;
    std::unordered_map<std::string, double> feat_score_mean() const;
    nb::dict feat_pair_importance() const;

    // Serialization
    nb::dict to_dict(const std::string& version) const;
    void from_dict(nb::dict model_dict);

private:
    DecisionTreeCPP plant_and_validate_single_tree(
        const double* X,
        const int32_t* y,
        size_t n_rows,
        size_t n_cols,
        const std::vector<std::vector<int32_t>>& class_indices,
        uint64_t seed
    ) const;
};

} // namespace rf_mc
