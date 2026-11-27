#include "tree.hpp"
#include <algorithm>
#include <map>
#include <numeric>
#include <cmath>
#include <tuple>

namespace rf_mc {

namespace {

constexpr double CAT_EPS = 1e-9;

double gini_counts(const std::vector<int32_t>& counts, int32_t n) {
    if (n <= 0) return 0.0;
    double sum_sq = 0.0;
    for (int32_t c : counts) {
        double p = static_cast<double>(c) / static_cast<double>(n);
        sum_sq += p * p;
    }
    return 1.0 - sum_sq;
}

int32_t get_int(const nb::dict& d, const char* key, int32_t fallback) {
    if (!d.contains(key) || d[key].is_none()) return fallback;
    return static_cast<int32_t>(nb::cast<double>(d[key]));
}

} // namespace

void DecisionTreeCPP::plant(
    const double* X,
    const int32_t* y,
    size_t n_rows,
    size_t n_cols,
    const std::vector<int32_t>& indices,
    std::vector<int32_t> feature_list,
    const std::vector<FeatType>& feat_types,
    const TreeParams& params
) {
    (void)n_rows;
    nodes.clear();
    used_features.clear();
    feature_sequence = feature_list;
    max_depth_reached = 0;
    tree_format = TREE_FORMAT;
    this->n_classes = params.n_classes;
    if (indices.empty() || feature_list.empty()) return;

    grow_tree(X, y, n_cols, indices, feature_list, feat_types, 1, params);
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
    leaf_node.n_samples = static_cast<int32_t>(indices.size());
    leaf_node.leaf_probs.assign(n_classes, 0.0);
    max_depth_reached = std::max(max_depth_reached, depth);

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

// v1.5.0 split rule: median with a '>' tie fallback and a special case for <= 2 samples.
SplitResult DecisionTreeCPP::split_legacy(
    const double* X,
    size_t n_cols,
    int32_t feat_idx,
    FeatType feat_type,
    const std::vector<int32_t>& indices
) const {
    SplitResult res;
    res.feat_type = feat_type;
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
            std::map<double, int32_t> counts;
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
                if (std::abs(v - mode_val) < CAT_EPS) res.idx_ge.push_back(idx);
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
    return res;
}

// Index into lower_counts of the cut: closest to the median by count or, with
// split_supervision >= 1, the lowest-Gini cut among those closest to the quartiles.
size_t DecisionTreeCPP::choose_cut(
    const std::vector<int32_t>& lower_counts,
    const std::vector<int32_t>& y_sorted,
    const TreeParams& params
) const {
    const double n = static_cast<double>(y_sorted.size());
    std::vector<double> targets = {n / 2.0};
    if (params.split_supervision > 0) {
        targets.push_back(n / 4.0);
        targets.push_back(3.0 * n / 4.0);
    }
    std::vector<size_t> cands;
    for (double t : targets) {
        size_t best = 0;
        double best_dist = std::abs(lower_counts[0] - t);
        for (size_t i = 1; i < lower_counts.size(); ++i) {
            double dist = std::abs(lower_counts[i] - t);
            if (dist < best_dist) {
                best_dist = dist;
                best = i;
            }
        }
        if (std::find(cands.begin(), cands.end(), best) == cands.end()) cands.push_back(best);
    }
    if (cands.size() == 1) return cands[0];

    size_t best = cands[0];
    double best_gini = INFINITY;
    for (size_t i : cands) {
        int32_t n_left = lower_counts[i];
        int32_t n_right = static_cast<int32_t>(y_sorted.size()) - n_left;
        std::vector<int32_t> left(n_classes, 0), right(n_classes, 0);
        for (int32_t r = 0; r < static_cast<int32_t>(y_sorted.size()); ++r) {
            if (r < n_left) left[y_sorted[r]]++;
            else right[y_sorted[r]]++;
        }
        double g = (n_left * gini_counts(left, n_left) + n_right * gini_counts(right, n_right)) / n;
        if (g < best_gini) {
            best_gini = g;
            best = i;
        }
    }
    return best;
}

double DecisionTreeCPP::split_gini(const int32_t* y, const SplitResult& res) const {
    std::vector<int32_t> ge(n_classes, 0), lt(n_classes, 0);
    for (int32_t idx : res.idx_ge) ge[y[idx]]++;
    for (int32_t idx : res.idx_lt) lt[y[idx]]++;
    int32_t n_ge = static_cast<int32_t>(res.idx_ge.size());
    int32_t n_lt = static_cast<int32_t>(res.idx_lt.size());
    return (n_ge * gini_counts(ge, n_ge) + n_lt * gini_counts(lt, n_lt)) / static_cast<double>(n_ge + n_lt);
}

// Split of the node rows on one feature. Every split is reproduced exactly by goes_ge().
SplitResult DecisionTreeCPP::find_split(
    const double* X,
    const int32_t* y,
    size_t n_cols,
    int32_t feat_idx,
    FeatType feat_type,
    const std::vector<int32_t>& indices,
    const TreeParams& params
) const {
    bool legacy = params.split_rule == 0 && params.split_supervision == 0;
    if (legacy && (feat_type == FeatType::NUMERIC || params.categorical_split == 0)) {
        return split_legacy(X, n_cols, feat_idx, feat_type, indices);
    }

    SplitResult res;
    res.feat_type = feat_type;
    const size_t n = indices.size();

    if (feat_type == FeatType::NUMERIC) {
        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return X[indices[a] * n_cols + feat_idx] < X[indices[b] * n_cols + feat_idx];
        });
        std::vector<double> sorted_vals(n);
        std::vector<int32_t> y_sorted(n);
        for (size_t i = 0; i < n; ++i) {
            sorted_vals[i] = X[indices[order[i]] * n_cols + feat_idx];
            y_sorted[i] = y[indices[order[i]]];
        }
        std::vector<size_t> bounds;
        std::vector<int32_t> lower_counts;
        for (size_t i = 0; i + 1 < n; ++i) {
            if (sorted_vals[i] != sorted_vals[i + 1]) {
                bounds.push_back(i);
                lower_counts.push_back(static_cast<int32_t>(i + 1));
            }
        }
        if (bounds.empty()) return res;
        size_t b = bounds[choose_cut(lower_counts, y_sorted, params)];
        double lo = sorted_vals[b], hi = sorted_vals[b + 1];
        double theta = lo + (hi - lo) / 2.0;
        if (!(lo < theta && theta <= hi)) theta = hi;
        res.split_val = theta;
        for (int32_t idx : indices) {
            if (X[idx * n_cols + feat_idx] >= theta) res.idx_ge.push_back(idx);
            else res.idx_lt.push_back(idx);
        }
        return res;
    }

    // Categorical: categories in ascending code order.
    std::map<double, int32_t> cat_pos;
    for (int32_t idx : indices) cat_pos.emplace(X[idx * n_cols + feat_idx], 0);
    if (cat_pos.size() < 2) return res;
    std::vector<double> cats;
    for (auto& kv : cat_pos) {
        kv.second = static_cast<int32_t>(cats.size());
        cats.push_back(kv.first);
    }
    std::vector<int32_t> inv(n);
    std::vector<int32_t> counts(cats.size(), 0);
    for (size_t i = 0; i < n; ++i) {
        inv[i] = cat_pos[X[indices[i] * n_cols + feat_idx]];
        counts[inv[i]]++;
    }

    if (params.categorical_split == 0) {
        size_t mode = 0;
        for (size_t c = 1; c < cats.size(); ++c) {
            if (counts[c] > counts[mode]) mode = c;
        }
        res.split_val = cats[mode];
        for (size_t i = 0; i < n; ++i) {
            if (inv[i] == static_cast<int32_t>(mode)) res.idx_ge.push_back(indices[i]);
            else res.idx_lt.push_back(indices[i]);
        }
        return res;
    }

    // Ordered categories: sort by class share, split at the count-median of that order.
    std::vector<std::vector<int32_t>> class_counts(cats.size(), std::vector<int32_t>(n_classes, 0));
    std::vector<int32_t> totals(n_classes, 0);
    for (size_t i = 0; i < n; ++i) {
        class_counts[inv[i]][y[indices[i]]]++;
        totals[y[indices[i]]]++;
    }
    int32_t target = 1;
    if (n_classes != 2) {
        target = static_cast<int32_t>(std::max_element(totals.begin(), totals.end()) - totals.begin());
    }
    std::vector<double> share(cats.size());
    for (size_t c = 0; c < cats.size(); ++c) {
        share[c] = static_cast<double>(class_counts[c][target]) / static_cast<double>(counts[c]);
    }
    std::vector<int32_t> cat_order(cats.size());
    std::iota(cat_order.begin(), cat_order.end(), 0);
    std::stable_sort(cat_order.begin(), cat_order.end(), [&](int32_t a, int32_t b) {
        return share[a] < share[b];
    });
    std::vector<int32_t> rank(cats.size());
    for (size_t r = 0; r < cat_order.size(); ++r) rank[cat_order[r]] = static_cast<int32_t>(r);

    std::vector<size_t> row_order(n);
    std::iota(row_order.begin(), row_order.end(), 0);
    std::stable_sort(row_order.begin(), row_order.end(), [&](size_t a, size_t b) {
        return rank[inv[a]] < rank[inv[b]];
    });
    std::vector<int32_t> sorted_rank(n), y_sorted(n);
    for (size_t i = 0; i < n; ++i) {
        sorted_rank[i] = rank[inv[row_order[i]]];
        y_sorted[i] = y[indices[row_order[i]]];
    }
    std::vector<size_t> bounds;
    std::vector<int32_t> lower_counts;
    for (size_t i = 0; i + 1 < n; ++i) {
        if (sorted_rank[i] != sorted_rank[i + 1]) {
            bounds.push_back(i);
            lower_counts.push_back(static_cast<int32_t>(i + 1));
        }
    }
    int32_t cut_rank = sorted_rank[bounds[choose_cut(lower_counts, y_sorted, params)]];
    for (size_t r = static_cast<size_t>(cut_rank) + 1; r < cat_order.size(); ++r) {
        res.split_set.push_back(cats[cat_order[r]]);
    }
    std::sort(res.split_set.begin(), res.split_set.end());
    res.feat_type = FeatType::CATEGORICAL_SET;
    for (size_t i = 0; i < n; ++i) {
        if (rank[inv[i]] > cut_rank) res.idx_ge.push_back(indices[i]);
        else res.idx_lt.push_back(indices[i]);
    }
    return res;
}

int32_t DecisionTreeCPP::grow_tree(
    const double* X,
    const int32_t* y,
    size_t n_cols,
    const std::vector<int32_t>& indices,
    std::vector<int32_t> F,
    const std::vector<FeatType>& feat_types,
    int32_t depth,
    const TreeParams& params
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
    if (depth >= params.max_depth || all_same) {
        return create_leaf(y, indices, depth);
    }

    const size_t min_child = static_cast<size_t>(std::max(1, params.min_samples_split));
    auto valid = [&](const SplitResult& r) {
        return r.idx_ge.size() >= min_child && r.idx_lt.size() >= min_child;
    };

    int32_t chosen_feat = -1;
    SplitResult split_res;

    if (params.split_supervision == 2) {
        // Best Gini split among the first `split_lookahead` features of the queue.
        double best_gini = INFINITY;
        size_t best_pos = 0;
        bool found = false;
        for (size_t pos = 0; pos < F.size(); ++pos) {
            SplitResult r = find_split(X, y, n_cols, F[pos], feat_types[F[pos]], indices, params);
            if (valid(r)) {
                double g = split_gini(y, r);
                if (!found || g < best_gini) {
                    best_gini = g;
                    best_pos = pos;
                    split_res = std::move(r);
                }
                found = true;
            }
            if (static_cast<int32_t>(pos) + 1 >= params.split_lookahead && found) break;
        }
        if (!found) return create_leaf(y, indices, depth);
        chosen_feat = F[best_pos];
        std::rotate(F.begin(), F.begin() + best_pos + 1, F.end());
    } else {
        for (size_t t = 0; t < F.size(); ++t) {
            int32_t feat = F[0];
            SplitResult r = find_split(X, y, n_cols, feat, feat_types[feat], indices, params);
            std::rotate(F.begin(), F.begin() + 1, F.end());
            if (valid(r)) {
                chosen_feat = feat;
                split_res = std::move(r);
                break;
            }
        }
        if (chosen_feat < 0) return create_leaf(y, indices, depth);
    }

    int32_t node_idx = static_cast<int32_t>(nodes.size());
    nodes.push_back(Node{}); // Allocate placeholder
    nodes[node_idx].feature_idx = chosen_feat;
    nodes[node_idx].feat_type = split_res.feat_type;
    nodes[node_idx].split_val = split_res.split_val;
    nodes[node_idx].split_set = split_res.split_set;
    nodes[node_idx].depth = depth;
    nodes[node_idx].n_ge = static_cast<int32_t>(split_res.idx_ge.size());
    nodes[node_idx].n_lt = static_cast<int32_t>(split_res.idx_lt.size());

    if (std::find(used_features.begin(), used_features.end(), chosen_feat) == used_features.end()) {
        used_features.push_back(chosen_feat);
    }

    int32_t left_child = grow_tree(X, y, n_cols, split_res.idx_ge, F, feat_types, depth + 1, params);
    int32_t right_child = grow_tree(X, y, n_cols, split_res.idx_lt, F, feat_types, depth + 1, params);

    nodes[node_idx].left_child = left_child;
    nodes[node_idx].right_child = right_child;
    return node_idx;
}

bool DecisionTreeCPP::goes_ge(const Node& node, double val) const {
    switch (node.feat_type) {
        case FeatType::NUMERIC:
            return val >= node.split_val;
        case FeatType::CATEGORICAL_SET:
            for (double s : node.split_set) {
                if (std::abs(val - s) < CAT_EPS) return true;
            }
            return false;
        default:
            return std::abs(val - node.split_val) < CAT_EPS;
    }
}

void DecisionTreeCPP::add_leaf(const Node& leaf, double weight, const InferParams& ip, std::vector<double>& out) const {
    if (ip.leaf_alpha > 0.0 && leaf.n_samples > 0) {
        double n = static_cast<double>(leaf.n_samples);
        double denom = n + n_classes * ip.leaf_alpha;
        for (int32_t c = 0; c < n_classes; ++c) {
            out[c] += weight * (leaf.leaf_probs[c] * n + ip.leaf_alpha) / denom;
        }
    } else {
        for (int32_t c = 0; c < n_classes; ++c) {
            out[c] += weight * leaf.leaf_probs[c];
        }
    }
}

// C4.5-style marginalization: a missing split sends the row down both branches,
// weighted by the training fraction of each branch.
void DecisionTreeCPP::accumulate_weighted(
    int32_t node_idx,
    const double* row,
    double weight,
    const InferParams& ip,
    std::vector<double>& out
) const {
    const Node& node = nodes[node_idx];
    if (node.is_leaf()) {
        add_leaf(node, weight, ip, out);
        return;
    }
    double val = row[node.feature_idx];
    if (std::isnan(val)) {
        double w_ge = 0.5, w_lt = 0.5;
        int32_t total = node.n_ge + node.n_lt;
        if (total > 0) {
            w_ge = static_cast<double>(node.n_ge) / total;
            w_lt = static_cast<double>(node.n_lt) / total;
        }
        accumulate_weighted(node.left_child, row, weight * w_ge, ip, out);
        accumulate_weighted(node.right_child, row, weight * w_lt, ip, out);
        return;
    }
    accumulate_weighted(goes_ge(node, val) ? node.left_child : node.right_child, row, weight, ip, out);
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
    collect_leaves(goes_ge(node, val) ? node.left_child : node.right_child, row, n_cols, leaf_nodes);
}

void DecisionTreeCPP::predict_row(const double* row, size_t n_cols, std::vector<double>& out_probs,
                                  const InferParams& ip) const {
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
        curr = goes_ge(node, val) ? node.left_child : node.right_child;
    }

    if (!had_nan) {
        add_leaf(nodes[curr], 1.0, ip, out_probs);
        return;
    }

    if (ip.missing_strategy == 1) {
        std::vector<const Node*> leaves;
        collect_leaves(0, row, n_cols, leaves);
        if (leaves.empty()) return;
        double w = 1.0 / static_cast<double>(leaves.size());
        for (const Node* leaf : leaves) add_leaf(*leaf, w, ip, out_probs);
    } else {
        accumulate_weighted(0, row, 1.0, ip, out_probs);
    }

    double total = 0.0;
    for (double p : out_probs) total += p;
    if (total > 0.0) {
        for (int32_t c = 0; c < n_classes; ++c) {
            out_probs[c] /= total;
        }
    }
}

int32_t DecisionTreeCPP::predict_class(const double* row, size_t n_cols, const InferParams& ip) const {
    std::vector<double> probs;
    predict_row(row, n_cols, probs, ip);
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
        if (node.n_samples > 0) d["n"] = node.n_samples;
        return d;
    }

    std::string feat_name = feature_names[node.feature_idx];
    nb::dict split_inner;
    if (node.feat_type == FeatType::CATEGORICAL_SET) {
        split_inner["feat_type"] = "categorical_set";
        nb::list set_list;
        for (double s : node.split_set) set_list.append(s);
        split_inner["split_val"] = set_list;
    } else {
        split_inner["feat_type"] = (node.feat_type == FeatType::NUMERIC) ? "numeric" : "categorical";
        split_inner["split_val"] = node.split_val;
    }
    if (node.n_ge + node.n_lt > 0) {
        split_inner["n_ge"] = node.n_ge;
        split_inner["n_lt"] = node.n_lt;
    }
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
    root["tree_format"] = tree_format;

    if (!feature_sequence.empty()) {
        nb::list seq_list;
        for (int32_t f : feature_sequence) {
            if (f >= 0 && f < static_cast<int32_t>(feature_names.size())) seq_list.append(feature_names[f].c_str());
        }
        root["feature_sequence"] = seq_list;
        root["selection_score"] = selection_score;
        root["attempts"] = attempts;
        root["hit_attempt_cap"] = hit_attempt_cap;
        root["max_depth_reached"] = max_depth_reached;
    } else {
        root["feature_sequence"] = nb::none();
        root["selection_score"] = nb::none();
        root["attempts"] = nb::none();
        root["hit_attempt_cap"] = false;
        root["max_depth_reached"] = nb::none();
    }
    return root;
}

int32_t DecisionTreeCPP::dict_to_node(
    nb::dict node_dict,
    const std::vector<std::string>& feature_names,
    const std::vector<std::string>& class_names,
    int32_t depth
) {
    if (node_dict.contains("leaf")) {
        int32_t leaf_idx = static_cast<int32_t>(nodes.size());
        Node leaf_node;
        leaf_node.feature_idx = -1;
        if (node_dict.contains("depth")) {
            std::string depth_str = nb::cast<std::string>(nb::str(node_dict["depth"]));
            size_t hash_pos = depth_str.find('#');
            leaf_node.depth = (hash_pos != std::string::npos) ? std::stoi(depth_str.substr(0, hash_pos)) : std::stoi(depth_str);
        } else {
            leaf_node.depth = depth;
        }
        leaf_node.n_samples = get_int(node_dict, "n", 0);
        leaf_node.leaf_probs.assign(class_names.size(), 0.0);
        max_depth_reached = std::max(max_depth_reached, leaf_node.depth);

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

    int32_t node_idx = static_cast<int32_t>(nodes.size());
    nodes.push_back(Node{});
    nodes[node_idx].feature_idx = feat_idx;
    nodes[node_idx].depth = depth;
    if (feat_type_str == "categorical_set") {
        nodes[node_idx].feat_type = FeatType::CATEGORICAL_SET;
        nb::list set_list = nb::cast<nb::list>(split_dict["split_val"]);
        for (size_t i = 0; i < set_list.size(); ++i) {
            nodes[node_idx].split_set.push_back(nb::cast<double>(set_list[i]));
        }
    } else {
        nodes[node_idx].feat_type = (feat_type_str == "numeric") ? FeatType::NUMERIC : FeatType::CATEGORICAL;
        nodes[node_idx].split_val = nb::cast<double>(split_dict["split_val"]);
    }
    nodes[node_idx].n_ge = get_int(split_dict, "n_ge", 0);
    nodes[node_idx].n_lt = get_int(split_dict, "n_lt", 0);

    if (std::find(used_features.begin(), used_features.end(), feat_idx) == used_features.end() && feat_idx >= 0) {
        used_features.push_back(feat_idx);
    }

    int32_t left_child = dict_to_node(nb::cast<nb::dict>(split_dict[">="]), feature_names, class_names, depth + 1);
    int32_t right_child = dict_to_node(nb::cast<nb::dict>(split_dict["<"]), feature_names, class_names, depth + 1);

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
    feature_sequence.clear();
    max_depth_reached = 0;
    n_classes = static_cast<int32_t>(class_names.size());
    if (tree_dict.contains("survived_score") && !tree_dict["survived_score"].is_none()) {
        survived_score = nb::cast<double>(tree_dict["survived_score"]);
    }
    if (tree_dict.contains("selection_score") && !tree_dict["selection_score"].is_none()) {
        selection_score = nb::cast<double>(tree_dict["selection_score"]);
    }
    tree_format = get_int(tree_dict, "tree_format", 1);
    attempts = get_int(tree_dict, "attempts", 0);
    hit_attempt_cap = tree_dict.contains("hit_attempt_cap") && !tree_dict["hit_attempt_cap"].is_none()
                      && nb::cast<bool>(tree_dict["hit_attempt_cap"]);
    if (tree_dict.contains("feature_sequence") && !tree_dict["feature_sequence"].is_none()) {
        nb::list seq = nb::cast<nb::list>(tree_dict["feature_sequence"]);
        for (size_t i = 0; i < seq.size(); ++i) {
            std::string name = nb::cast<std::string>(seq[i]);
            for (size_t f = 0; f < feature_names.size(); ++f) {
                if (feature_names[f] == name) {
                    feature_sequence.push_back(static_cast<int32_t>(f));
                    break;
                }
            }
        }
    }
    if (tree_dict.contains("data")) {
        nb::dict data = nb::cast<nb::dict>(tree_dict["data"]);
        if (data.size() > 0) dict_to_node(data, feature_names, class_names, 1);
    }
}

} // namespace rf_mc
