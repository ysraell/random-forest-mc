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

// One Monte Carlo candidate tree (kept for feature importance and ensemble selection).
struct CandidateRecord {
    int32_t slot = 0;
    int32_t batch = 0;
    std::vector<int32_t> F;
    std::vector<int32_t> used;
    int32_t depth = 0;
    double s_sel = 0.0;
    double s_score = 0.0;
    bool has_score = false;
    bool accepted = false;
};

struct SlotResult {
    DecisionTreeCPP tree;
    std::vector<CandidateRecord> records;
    std::vector<DecisionTreeCPP> cand_trees;  // selection == ensemble only
    std::vector<float> cand_probs;            // (n_cand, n_ens, K), selection == ensemble only
};

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

    // v1.6.0 (batch sizes, floor and caps are resolved by the Python wrapper).
    int32_t n_train_pclass = 10;
    int32_t n_sel_pclass = 10;
    int32_t n_score_pclass = 0;
    int32_t ens_sel_pclass = 0;
    int32_t ens_score_pclass = 0;
    double tau_min = -1e300;
    int32_t max_attempts = 0;            // <= 0: unlimited
    int32_t feature_max_sampling = -1;   // upper bound of k (effective-depth sampling), -1: max_feature
    int32_t selection = 0;               // 0: per tree, 1: ensemble library
    TreeParams tree_params;
    InferParams infer_params;
    int32_t tree_weighting = 0;          // 0: survival score, 1: SAMME
    std::vector<double> class_priors;    // empty: no prior correction

    std::vector<DecisionTreeCPP> trees;
    std::vector<double> survived_scores;
    std::vector<std::string> feature_names;
    std::vector<std::string> class_names;
    std::vector<FeatType> feature_types;

    std::vector<CandidateRecord> candidate_log;
    std::vector<DecisionTreeCPP> candidate_trees;
    std::vector<float> candidate_probs;
    std::vector<int32_t> ens_sel_idx;
    std::vector<int32_t> ens_score_idx;

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

    // Forest from candidate indices (with repetition), after ensemble selection.
    void set_trees_from_candidates(const std::vector<int32_t>& chosen);

    std::vector<double> tree_weights() const;

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

    // Per-tree probabilities (n_trees, n_rows, K), no voting or priors.
    void trees_proba_batch(
        const double* X,
        size_t n_rows,
        size_t n_cols,
        std::vector<double>& out_flat
    ) const;

    // Feature analysis
    std::unordered_map<std::string, double> feat_importance() const;
    std::unordered_map<std::string, double> feat_score_mean() const;
    nb::dict feat_pair_importance() const;
    nb::list candidate_log_list() const;

    // Serialization
    nb::dict to_dict(const std::string& version) const;
    void from_dict(nb::dict model_dict);

private:
    SlotResult run_slot(
        const double* X,
        const int32_t* y,
        size_t n_rows,
        size_t n_cols,
        const std::vector<std::vector<int32_t>>& class_indices,
        uint64_t seed,
        int32_t slot
    ) const;
};

} // namespace rf_mc
