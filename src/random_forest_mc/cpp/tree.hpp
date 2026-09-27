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

class DecisionTreeCPP {
public:
    std::vector<Node> nodes;
    double survived_score = 0.0;
    std::vector<int32_t> used_features;
    int32_t n_classes = 0;

    DecisionTreeCPP() = default;
    explicit DecisionTreeCPP(int32_t n_classes) : n_classes(n_classes) {}

    // In-place row prediction
    void predict_row(const double* row, size_t n_cols, std::vector<double>& out_probs) const;

    // Highest probability class for a single row
    int32_t predict_class(const double* row, size_t n_cols) const;

    // Plant / grow tree from scratch
    void plant(
        const double* X,
        const int32_t* y,
        size_t n_rows,
        size_t n_cols,
        const std::vector<int32_t>& indices,
        std::vector<int32_t> feature_list,
        const std::vector<FeatType>& feat_types,
        int32_t max_depth,
        int32_t min_samples_split,
        int32_t n_classes
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
        size_t n_rows,
        size_t n_cols,
        const std::vector<int32_t>& indices,
        std::vector<int32_t> F,
        const std::vector<FeatType>& feat_types,
        int32_t depth,
        int32_t max_depth,
        int32_t min_samples_split
    );

    int32_t create_leaf(
        const int32_t* y,
        const std::vector<int32_t>& indices,
        int32_t depth
    );

    SplitResult split_data(
        const double* X,
        size_t n_cols,
        int32_t feat_idx,
        FeatType feat_type,
        const std::vector<int32_t>& indices
    );

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
        const std::vector<std::string>& class_names
    );
};

} // namespace rf_mc
