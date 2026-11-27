#pragma once

#include "node.hpp"
#include "types.hpp"
#include <vector>
#include <string>
#include <cstdint>
#include <random>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace rf_mc {

constexpr int32_t TREE_FORMAT = 2;

class DecisionTreeCPP {
public:
    std::vector<Node> nodes;
    double survived_score = 0.0;
    double selection_score = 0.0;
    std::vector<int32_t> used_features;
    std::vector<int32_t> feature_sequence;
    int32_t n_classes = 0;
    int32_t attempts = 0;
    bool hit_attempt_cap = false;
    int32_t max_depth_reached = 0;
    int32_t tree_format = TREE_FORMAT;

    DecisionTreeCPP() = default;
    explicit DecisionTreeCPP(int32_t n_classes) : n_classes(n_classes) {}

    // In-place row prediction (NaN marks a missing feature)
    void predict_row(const double* row, size_t n_cols, std::vector<double>& out_probs,
                     const InferParams& ip = InferParams{}) const;

    // Highest probability class for a single row
    int32_t predict_class(const double* row, size_t n_cols, const InferParams& ip = InferParams{}) const;

    // Plant / grow tree from scratch
    void plant(
        const double* X,
        const int32_t* y,
        size_t n_rows,
        size_t n_cols,
        const std::vector<int32_t>& indices,
        std::vector<int32_t> feature_list,
        const std::vector<FeatType>& feat_types,
        const TreeParams& params
    );

    // Serialization to/from Python dict
    nb::dict to_dict(
        const std::vector<std::string>& feature_names,
        const std::vector<std::string>& class_names,
        const std::string& module_version
    ) const;

    void from_dict(
        nb::dict tree_dict,
        const std::vector<std::string>& feature_names,
        const std::vector<std::string>& class_names
    );

private:
    int32_t grow_tree(
        const double* X,
        const int32_t* y,
        size_t n_cols,
        const std::vector<int32_t>& indices,
        std::vector<int32_t> F,
        const std::vector<FeatType>& feat_types,
        int32_t depth,
        const TreeParams& params
    );

    int32_t create_leaf(
        const int32_t* y,
        const std::vector<int32_t>& indices,
        int32_t depth
    );

    SplitResult split_legacy(
        const double* X,
        size_t n_cols,
        int32_t feat_idx,
        FeatType feat_type,
        const std::vector<int32_t>& indices
    ) const;

    SplitResult find_split(
        const double* X,
        const int32_t* y,
        size_t n_cols,
        int32_t feat_idx,
        FeatType feat_type,
        const std::vector<int32_t>& indices,
        const TreeParams& params
    ) const;

    size_t choose_cut(
        const std::vector<int32_t>& lower_counts,
        const std::vector<int32_t>& y_sorted,
        const TreeParams& params
    ) const;

    double split_gini(const int32_t* y, const SplitResult& res) const;

    bool goes_ge(const Node& node, double val) const;

    void add_leaf(const Node& leaf, double weight, const InferParams& ip, std::vector<double>& out) const;

    void accumulate_weighted(
        int32_t node_idx,
        const double* row,
        double weight,
        const InferParams& ip,
        std::vector<double>& out
    ) const;

    void collect_leaves(
        int32_t node_idx,
        const double* row,
        size_t n_cols,
        std::vector<const Node*>& leaf_nodes
    ) const;

    nb::dict node_to_dict(
        int32_t node_idx,
        const std::vector<std::string>& feature_names,
        const std::vector<std::string>& class_names
    ) const;

    int32_t dict_to_node(
        nb::dict node_dict,
        const std::vector<std::string>& feature_names,
        const std::vector<std::string>& class_names,
        int32_t depth
    );
};

} // namespace rf_mc
