#include "tree.hpp"
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <cmath>

namespace rf_mc {

void DecisionTreeCPP::plant(
    const double* X,
    const int32_t* y,
    size_t n_rows,
    size_t n_cols,
    const std::vector<int32_t>& indices,
    std::vector<int32_t> feature_list,
    const std::vector<FeatType>& feat_types,
    int32_t max_depth,
    int32_t min_samples_split,
    int32_t num_classes
) {
    nodes.clear();
    used_features.clear();
    this->n_classes = num_classes;
    if (indices.empty() || feature_list.empty()) return;

    grow_tree(X, y, n_rows, n_cols, indices, feature_list, feat_types, 1, max_depth, min_samples_split);
}

int32_t DecisionTreeCPP::create_leaf(
    const int32_t* y,
    const std::vector<int32_t>& indices,
    int32_t depth
) {
    int32_t leaf_idx = static_cast<int32_t>(nodes.size());
    Node leaf_node;
    leaf_node.feature_idx = -1;
    leaf_node.depth = depth;
    leaf_node.leaf_probs.assign(n_classes, 0.0);

    if (!indices.empty()) {
        double inv_total = 1.0 / static_cast<double>(indices.size());
        for (int32_t idx : indices) {
            int32_t c = y[idx];
            if (c >= 0 && c < n_classes) {
                leaf_node.leaf_probs[c] += inv_total;
            }
        }
    }
    nodes.push_back(std::move(leaf_node));
    return leaf_idx;
}

SplitResult DecisionTreeCPP::split_data(
    const double* X,
    size_t n_cols,
    int32_t feat_idx,
    FeatType feat_type,
    const std::vector<int32_t>& indices
) {
    SplitResult res;
    if (indices.size() > 2) {
        if (feat_type == FeatType::NUMERIC) {
            std::vector<double> vals;
            vals.reserve(indices.size());
            for (int32_t idx : indices) {
                vals.push_back(X[idx * n_cols + feat_idx]);
            }
            size_t n = vals.size();
            double split_val;
            if (n % 2 == 1) {
                std::nth_element(vals.begin(), vals.begin() + n / 2, vals.end());
                split_val = vals[n / 2];
            } else {
                std::nth_element(vals.begin(), vals.begin() + n / 2, vals.end());
                double mid2 = vals[n / 2];
                std::nth_element(vals.begin(), vals.begin() + n / 2 - 1, vals.begin() + n / 2);
                double mid1 = vals[n / 2 - 1];
                split_val = (mid1 + mid2) / 2.0;
            }

            res.split_val = split_val;
            for (int32_t idx : indices) {
                double v = X[idx * n_cols + feat_idx];
                if (v >= split_val) res.idx_ge.push_back(idx);
                else res.idx_lt.push_back(idx);
            }

            if (res.idx_ge.empty() || res.idx_lt.empty()) {
                res.idx_ge.clear();
                res.idx_lt.clear();
                for (int32_t idx : indices) {
                    double v = X[idx * n_cols + feat_idx];
                    if (v > split_val) res.idx_ge.push_back(idx);
                    else res.idx_lt.push_back(idx);
                }
            }
        } else {
            // Categorical mode
            std::unordered_map<double, int32_t> counts;
            for (int32_t idx : indices) {
                counts[X[idx * n_cols + feat_idx]]++;
            }
            double mode_val = 0.0;
            int32_t max_count = -1;
            for (const auto& pair : counts) {
                if (pair.second > max_count) {
                    max_count = pair.second;
                    mode_val = pair.first;
                }
            }
            res.split_val = mode_val;
            for (int32_t idx : indices) {
                double v = X[idx * n_cols + feat_idx];
                if (std::abs(v - mode_val) < 1e-9) res.idx_ge.push_back(idx);
                else res.idx_lt.push_back(idx);
            }
        }
    } else {
        // 2 or fewer samples: sort by feature value
        std::vector<int32_t> sorted_idx = indices;
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&](int32_t a, int32_t b) {
            return X[a * n_cols + feat_idx] < X[b * n_cols + feat_idx];
        });
        if (sorted_idx.size() == 2) {
            res.idx_ge = {sorted_idx[1]};
            res.idx_lt = {sorted_idx[0]};
            res.split_val = X[sorted_idx[1] * n_cols + feat_idx];
        } else {
            res.idx_ge = sorted_idx;
            res.split_val = X[sorted_idx[0] * n_cols + feat_idx];
        }
    }
    res.valid = (!res.idx_ge.empty() && !res.idx_lt.empty());
    return res;
}

int32_t DecisionTreeCPP::grow_tree(
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
) {
    if (indices.empty() || F.empty()) {
        return create_leaf(y, indices, depth);
    }

    bool all_same = true;
    int32_t first_class = y[indices[0]];
    for (size_t i = 1; i < indices.size(); ++i) {
        if (y[indices[i]] != first_class) {
            all_same = false;
            break;
        }
    }
    if (depth >= max_depth || all_same) {
        return create_leaf(y, indices, depth);
    }

    int32_t first_feat = F[0];
    bool pass = false;
    int32_t chosen_feat = -1;
    SplitResult split_res;

    while (!pass) {
        int32_t feat = F[0];
        split_res = split_data(X, n_cols, feat, feat_types[feat], indices);
        F.erase(F.begin());
        F.push_back(feat);

        if (static_cast<int32_t>(split_res.idx_ge.size()) >= min_samples_split &&
            static_cast<int32_t>(split_res.idx_lt.size()) >= min_samples_split) {
            pass = true;
            chosen_feat = feat;
        } else if (first_feat == F[0]) {
            return create_leaf(y, indices, depth);
        }
    }

    int32_t node_idx = static_cast<int32_t>(nodes.size());
    nodes.push_back(Node{}); // Allocate placeholder
    nodes[node_idx].feature_idx = chosen_feat;
    nodes[node_idx].feat_type = feat_types[chosen_feat];
    nodes[node_idx].split_val = split_res.split_val;
    nodes[node_idx].depth = depth;

    if (std::find(used_features.begin(), used_features.end(), chosen_feat) == used_features.end()) {
        used_features.push_back(chosen_feat);
    }

    int32_t left_child = grow_tree(X, y, n_rows, n_cols, split_res.idx_ge, F, feat_types, depth + 1, max_depth, min_samples_split);
    int32_t right_child = grow_tree(X, y, n_rows, n_cols, split_res.idx_lt, F, feat_types, depth + 1, max_depth, min_samples_split);

    nodes[node_idx].left_child = left_child;
    nodes[node_idx].right_child = right_child;
    return node_idx;
}

void DecisionTreeCPP::collect_leaves(
    int32_t node_idx,
    const double* row,
    size_t n_cols,
    std::vector<const Node*>& leaf_nodes
) const {
    const Node& node = nodes[node_idx];
    if (node.is_leaf()) {
        leaf_nodes.push_back(&node);
        return;
    }
    double val = row[node.feature_idx];
    if (std::isnan(val)) {
        collect_leaves(node.left_child, row, n_cols, leaf_nodes);
        collect_leaves(node.right_child, row, n_cols, leaf_nodes);
        return;
    }
    if (node.feat_type == FeatType::NUMERIC) {
        if (val >= node.split_val) {
            collect_leaves(node.left_child, row, n_cols, leaf_nodes);
        } else {
            collect_leaves(node.right_child, row, n_cols, leaf_nodes);
        }
    } else {
        if (std::abs(val - node.split_val) < 1e-9) {
            collect_leaves(node.left_child, row, n_cols, leaf_nodes);
        } else {
            collect_leaves(node.right_child, row, n_cols, leaf_nodes);
        }
    }
}

void DecisionTreeCPP::predict_row(const double* row, size_t n_cols, std::vector<double>& out_probs) const {
    out_probs.assign(n_classes, 0.0);
    if (nodes.empty()) return;

    int32_t curr = 0;
    bool had_nan = false;
    while (!nodes[curr].is_leaf()) {
        const Node& node = nodes[curr];
        double val = row[node.feature_idx];
        if (std::isnan(val)) {
            had_nan = true;
            break;
        }
        if (node.feat_type == FeatType::NUMERIC) {
            curr = (val >= node.split_val) ? node.left_child : node.right_child;
        } else {
            curr = (std::abs(val - node.split_val) < 1e-9) ? node.left_child : node.right_child;
        }
    }

    if (!had_nan) {
        out_probs = nodes[curr].leaf_probs;
        return;
    }

    std::vector<const Node*> leaves;
    collect_leaves(0, row, n_cols, leaves);
    if (leaves.empty()) return;

    double n_leaves = static_cast<double>(leaves.size());
    for (const Node* leaf : leaves) {
        for (int32_t c = 0; c < n_classes; ++c) {
            out_probs[c] += leaf->leaf_probs[c];
        }
    }
    for (int32_t c = 0; c < n_classes; ++c) {
        out_probs[c] /= n_leaves;
    }
    double total = 0.0;
    for (double p : out_probs) total += p;
    if (total > 0.0) {
        for (int32_t c = 0; c < n_classes; ++c) {
            out_probs[c] /= total;
        }
    }
}

int32_t DecisionTreeCPP::predict_class(const double* row, size_t n_cols) const {
    std::vector<double> probs;
    predict_row(row, n_cols, probs);
    int32_t best_c = 0;
    double best_p = -1.0;
    for (int32_t c = 0; c < static_cast<int32_t>(probs.size()); ++c) {
        if (probs[c] > best_p) {
            best_p = probs[c];
            best_c = c;
        }
    }
    return best_c;
}

nb::dict DecisionTreeCPP::node_to_dict(
    int32_t node_idx,
    const std::vector<std::string>& feature_names,
    const std::vector<std::string>& class_names
) const {
    const Node& node = nodes[node_idx];
    nb::dict d;
    if (node.is_leaf()) {
        nb::dict leaf_dict;
        for (size_t c = 0; c < class_names.size() && c < node.leaf_probs.size(); ++c) {
            if (node.leaf_probs[c] > 0.0) {
                leaf_dict[class_names[c].c_str()] = node.leaf_probs[c];
            }
        }
        d["leaf"] = leaf_dict;
        d["depth"] = (std::to_string(node.depth) + "#").c_str();
        return d;
    }

    std::string feat_name = feature_names[node.feature_idx];
    nb::dict split_inner;
    split_inner["feat_type"] = (node.feat_type == FeatType::NUMERIC) ? "numeric" : "categorical";
    split_inner["split_val"] = node.split_val;
    split_inner[">="] = node_to_dict(node.left_child, feature_names, class_names);
    split_inner["<"] = node_to_dict(node.right_child, feature_names, class_names);

    nb::dict feat_dict;
    feat_dict["split"] = split_inner;
    d[feat_name.c_str()] = feat_dict;
    return d;
}

nb::dict DecisionTreeCPP::to_dict(
    const std::vector<std::string>& feature_names,
    const std::vector<std::string>& class_names,
    const std::string& module_version
) const {
    nb::dict root;
    if (!nodes.empty()) {
        root["data"] = node_to_dict(0, feature_names, class_names);
    } else {
        root["data"] = nb::dict();
    }
    nb::list class_list;
    for (const auto& c : class_names) class_list.append(c.c_str());
    root["class_vals"] = class_list;
    root["survived_score"] = survived_score;

    nb::list feats_list;
    for (const auto& f : feature_names) feats_list.append(f.c_str());
    root["features"] = feats_list;

    nb::list used_list;
    for (int32_t uf : used_features) {
        if (uf >= 0 && uf < static_cast<int32_t>(feature_names.size())) {
            used_list.append(feature_names[uf].c_str());
        }
    }
    root["used_features"] = used_list;
    root["module_version"] = module_version.c_str();
    return root;
}

int32_t DecisionTreeCPP::dict_to_node(
    nb::dict node_dict,
    const std::vector<std::string>& feature_names,
    const std::vector<std::string>& class_names
) {
    if (node_dict.contains("leaf")) {
        int32_t leaf_idx = static_cast<int32_t>(nodes.size());
        Node leaf_node;
        leaf_node.feature_idx = -1;
        std::string depth_str = nb::cast<std::string>(node_dict["depth"]);
        size_t hash_pos = depth_str.find('#');
        leaf_node.depth = (hash_pos != std::string::npos) ? std::stoi(depth_str.substr(0, hash_pos)) : std::stoi(depth_str);
        leaf_node.leaf_probs.assign(class_names.size(), 0.0);

        nb::dict leaf_probs_dict = nb::cast<nb::dict>(node_dict["leaf"]);
        for (size_t c = 0; c < class_names.size(); ++c) {
            const auto& c_name = class_names[c];
            if (leaf_probs_dict.contains(c_name.c_str())) {
                leaf_node.leaf_probs[c] = nb::cast<double>(leaf_probs_dict[c_name.c_str()]);
            }
        }
        nodes.push_back(std::move(leaf_node));
        return leaf_idx;
    }

    // Split node: key is feature name
    std::string feat_name;
    for (auto item : node_dict) {
        feat_name = nb::cast<std::string>(item.first);
        break;
    }
    int32_t feat_idx = -1;
    for (size_t i = 0; i < feature_names.size(); ++i) {
        if (feature_names[i] == feat_name) {
            feat_idx = static_cast<int32_t>(i);
            break;
        }
    }

    nb::dict feat_dict = nb::cast<nb::dict>(node_dict[feat_name.c_str()]);
    nb::dict split_dict = nb::cast<nb::dict>(feat_dict["split"]);
    std::string feat_type_str = nb::cast<std::string>(split_dict["feat_type"]);
    double split_val = nb::cast<double>(split_dict["split_val"]);

    int32_t node_idx = static_cast<int32_t>(nodes.size());
    nodes.push_back(Node{});
    nodes[node_idx].feature_idx = feat_idx;
    nodes[node_idx].feat_type = (feat_type_str == "numeric") ? FeatType::NUMERIC : FeatType::CATEGORICAL;
    nodes[node_idx].split_val = split_val;

    if (std::find(used_features.begin(), used_features.end(), feat_idx) == used_features.end() && feat_idx >= 0) {
        used_features.push_back(feat_idx);
    }

    int32_t left_child = dict_to_node(nb::cast<nb::dict>(split_dict[">="]), feature_names, class_names);
    int32_t right_child = dict_to_node(nb::cast<nb::dict>(split_dict["<"]), feature_names, class_names);

    nodes[node_idx].left_child = left_child;
    nodes[node_idx].right_child = right_child;
    return node_idx;
}

void DecisionTreeCPP::from_dict(
    nb::dict tree_dict,
    const std::vector<std::string>& feature_names,
    const std::vector<std::string>& class_names
) {
    nodes.clear();
    used_features.clear();
    n_classes = static_cast<int32_t>(class_names.size());
    if (tree_dict.contains("survived_score")) {
        survived_score = nb::cast<double>(tree_dict["survived_score"]);
    }
    if (tree_dict.contains("data")) {
        dict_to_node(nb::cast<nb::dict>(tree_dict["data"]), feature_names, class_names);
    }
}

} // namespace rf_mc
