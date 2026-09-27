#include "forest.hpp"
#include <algorithm>
#include <random>
#include <numeric>
#include <thread>
#include <atomic>
#include <map>
#include <cmath>

namespace rf_mc {

void RandomForestCPP::set_features_and_classes(
    const std::vector<std::string>& f_names,
    const std::vector<std::string>& c_names,
    const std::vector<std::string>& f_types
) {
    feature_names = f_names;
    class_names = c_names;
    feature_types.clear();
    for (const auto& t : f_types) {
        feature_types.push_back((t == "numeric") ? FeatType::NUMERIC : FeatType::CATEGORICAL);
    }
}

DecisionTreeCPP RandomForestCPP::plant_and_validate_single_tree(
    const double* X,
    const int32_t* y,
    size_t n_rows,
    size_t n_cols,
    const std::vector<std::vector<int32_t>>& class_indices,
    uint64_t seed
) const {
    std::mt19937_64 rng(seed);

    // Compute N = min(batch_train + batch_val, min_class_count)
    size_t min_class_count = 1000000000;
    for (const auto& c_idx : class_indices) {
        if (!c_idx.empty()) {
            min_class_count = std::min(min_class_count, c_idx.size());
        }
    }
    size_t target_N = static_cast<size_t>(batch_train_pclass + batch_val_pclass);
    size_t N = std::min(target_N, min_class_count);

    size_t n_train_per_c = std::min(static_cast<size_t>(batch_train_pclass), N);

    std::vector<int32_t> idx_train;
    std::vector<int32_t> idx_val;

    for (const auto& c_idx : class_indices) {
        if (c_idx.empty()) continue;
        std::vector<int32_t> pool = c_idx;
        std::shuffle(pool.begin(), pool.end(), rng);
        for (size_t i = 0; i < N && i < pool.size(); ++i) {
            if (i < n_train_per_c) {
                idx_train.push_back(pool[i]);
            } else {
                idx_val.push_back(pool[i]);
            }
        }
    }

    // Monte Carlo tree planting & validation loop
    double threshold = th_start;
    int32_t discarded = 0;
    DecisionTreeCPP max_tree(static_cast<int32_t>(class_names.size()));
    double max_th_val = 0.0;
    DecisionTreeCPP current_tree(static_cast<int32_t>(class_names.size()));

    int32_t min_f = (min_feature > 0) ? min_feature : 2;
    int32_t max_f = (max_feature > 0) ? max_feature : static_cast<int32_t>(feature_names.size());
    min_f = std::min(min_f, static_cast<int32_t>(feature_names.size()));
    max_f = std::min(max_f, static_cast<int32_t>(feature_names.size()));
    if (min_f > max_f) min_f = max_f;

    std::vector<int32_t> all_feats(feature_names.size());
    std::iota(all_feats.begin(), all_feats.end(), 0);

    while (true) {
        // Sample feature subset
        std::uniform_int_distribution<int32_t> feat_count_dist(min_f, max_f);
        int32_t k = feat_count_dist(rng);
        std::vector<int32_t> shuffled_feats = all_feats;
        std::shuffle(shuffled_feats.begin(), shuffled_feats.end(), rng);
        std::vector<int32_t> sample_f(shuffled_feats.begin(), shuffled_feats.begin() + k);

        if (temporal_features) {
            // Sort by integer suffix if applicable
            std::sort(sample_f.begin(), sample_f.end(), [&](int32_t a, int32_t b) {
                const auto& na = feature_names[a];
                const auto& nb = feature_names[b];
                size_t pos_a = na.rfind('_');
                size_t pos_b = nb.rfind('_');
                if (pos_a != std::string::npos && pos_b != std::string::npos) {
                    try {
                        return std::stoi(na.substr(pos_a + 1)) < std::stoi(nb.substr(pos_b + 1));
                    } catch (...) {}
                }
                return a < b;
            });
        }

        current_tree.plant(
            X, y, n_rows, n_cols,
            idx_train, sample_f, feature_types,
            max_depth, min_samples_split,
            static_cast<int32_t>(class_names.size())
        );

        // Validation accuracy on idx_val
        double th_val = 0.0;
        if (!idx_val.empty()) {
            int32_t correct = 0;
            for (int32_t v_idx : idx_val) {
                int32_t pred_c = current_tree.predict_class(X + v_idx * n_cols, n_cols);
                if (pred_c == y[v_idx]) {
                    correct++;
                }
            }
            th_val = static_cast<double>(correct) / static_cast<double>(idx_val.size());
        } else {
            th_val = 1.0;
        }

        if (th_val < threshold) {
            if (get_best_tree && th_val > max_th_val) {
                max_tree = current_tree;
                max_th_val = th_val;
            }
            discarded++;
            if (discarded >= max_discard_trees) {
                if (get_best_tree) {
                    if (max_th_val > 0.0 || current_tree.nodes.empty()) {
                        current_tree = max_tree;
                    }
                    max_th_val = std::max(max_th_val, th_val);
                    break;
                } else {
                    threshold -= delta_th;
                }
            }
        } else {
            max_th_val = th_val;
            break;
        }
    }

    current_tree.survived_score = max_th_val;
    return current_tree;
}

void RandomForestCPP::fit(
    const double* X,
    const int32_t* y,
    size_t n_rows,
    size_t n_cols,
    int32_t n_threads
) {
    trees.clear();
    survived_scores.clear();
    if (n_rows == 0 || n_cols == 0 || class_names.empty()) return;

    // Group sample indices by class
    std::vector<std::vector<int32_t>> class_indices(class_names.size());
    for (size_t i = 0; i < n_rows; ++i) {
        int32_t c = y[i];
        if (c >= 0 && c < static_cast<int32_t>(class_names.size())) {
            class_indices[c].push_back(static_cast<int32_t>(i));
        }
    }

    int32_t threads_to_use = (n_threads <= 0) ? static_cast<int32_t>(std::thread::hardware_concurrency()) : n_threads;
    if (threads_to_use <= 0) threads_to_use = 1;
    threads_to_use = std::min(threads_to_use, n_trees);

    std::vector<DecisionTreeCPP> planted(n_trees);
    std::atomic<int32_t> next_tree(0);
    std::vector<std::thread> workers;
    workers.reserve(threads_to_use);

    for (int32_t t = 0; t < threads_to_use; ++t) {
        workers.emplace_back([&, t]() {
            while (true) {
                int32_t tree_idx = next_tree.fetch_add(1);
                if (tree_idx >= n_trees) break;
                planted[tree_idx] = plant_and_validate_single_tree(
                    X, y, n_rows, n_cols, class_indices, random_seed + tree_idx * 10007 + t
                );
            }
        });
    }

    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }

    trees = std::move(planted);
    survived_scores.resize(trees.size());
    for (size_t i = 0; i < trees.size(); ++i) {
        survived_scores[i] = trees[i].survived_score;
    }
}

void RandomForestCPP::predict_row_proba(
    const double* row,
    size_t n_cols,
    std::vector<double>& out_probs
) const {
    size_t num_classes = class_names.size();
    out_probs.assign(num_classes, 0.0);
    if (trees.empty()) return;

    std::vector<double> tree_p(num_classes, 0.0);
    double score_sum = 0.0;
    for (double s : survived_scores) score_sum += s;
    if (score_sum <= 0.0) score_sum = 1.0;

    if (soft_voting) {
        if (weighted_tree) {
            for (size_t i = 0; i < trees.size(); ++i) {
                trees[i].predict_row(row, n_cols, tree_p);
                double w = survived_scores[i];
                for (size_t c = 0; c < num_classes; ++c) {
                    out_probs[c] += tree_p[c] * w;
                }
            }
            for (size_t c = 0; c < num_classes; ++c) {
                out_probs[c] /= score_sum;
            }
        } else {
            for (size_t i = 0; i < trees.size(); ++i) {
                trees[i].predict_row(row, n_cols, tree_p);
                for (size_t c = 0; c < num_classes; ++c) {
                    out_probs[c] += tree_p[c];
                }
            }
            double inv_n = 1.0 / static_cast<double>(trees.size());
            for (size_t c = 0; c < num_classes; ++c) {
                out_probs[c] *= inv_n;
            }
        }
    } else {
        // Hard voting
        if (weighted_tree) {
            for (size_t i = 0; i < trees.size(); ++i) {
                int32_t best_c = trees[i].predict_class(row, n_cols);
                out_probs[best_c] += survived_scores[i];
            }
            for (size_t c = 0; c < num_classes; ++c) {
                out_probs[c] /= score_sum;
            }
        } else {
            for (size_t i = 0; i < trees.size(); ++i) {
                int32_t best_c = trees[i].predict_class(row, n_cols);
                out_probs[best_c] += 1.0;
            }
            double inv_n = 1.0 / static_cast<double>(trees.size());
            for (size_t c = 0; c < num_classes; ++c) {
                out_probs[c] *= inv_n;
            }
        }
    }
}

int32_t RandomForestCPP::predict_row_class(
    const double* row,
    size_t n_cols
) const {
    std::vector<double> probs;
    predict_row_proba(row, n_cols, probs);
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

void RandomForestCPP::predict_batch(
    const double* X,
    size_t n_rows,
    size_t n_cols,
    std::vector<int32_t>& out_classes,
    int32_t n_threads
) const {
    out_classes.resize(n_rows);
    int32_t threads_to_use = (n_threads <= 0) ? static_cast<int32_t>(std::thread::hardware_concurrency()) : n_threads;
    if (threads_to_use <= 0) threads_to_use = 1;
    threads_to_use = std::min(threads_to_use, static_cast<int32_t>(n_rows));

    std::atomic<size_t> row_counter(0);
    std::vector<std::thread> workers;
    workers.reserve(threads_to_use);

    for (int32_t t = 0; t < threads_to_use; ++t) {
        workers.emplace_back([&]() {
            const size_t batch = 128;
            while (true) {
                size_t start = row_counter.fetch_add(batch);
                if (start >= n_rows) break;
                size_t end = std::min(start + batch, n_rows);
                for (size_t r = start; r < end; ++r) {
                    out_classes[r] = predict_row_class(X + r * n_cols, n_cols);
                }
            }
        });
    }

    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }
}

void RandomForestCPP::predict_proba_batch(
    const double* X,
    size_t n_rows,
    size_t n_cols,
    std::vector<double>& out_probs_flat,
    int32_t n_threads
) const {
    size_t num_classes = class_names.size();
    out_probs_flat.resize(n_rows * num_classes, 0.0);

    int32_t threads_to_use = (n_threads <= 0) ? static_cast<int32_t>(std::thread::hardware_concurrency()) : n_threads;
    if (threads_to_use <= 0) threads_to_use = 1;
    threads_to_use = std::min(threads_to_use, static_cast<int32_t>(n_rows));

    std::atomic<size_t> row_counter(0);
    std::vector<std::thread> workers;
    workers.reserve(threads_to_use);

    for (int32_t t = 0; t < threads_to_use; ++t) {
        workers.emplace_back([&]() {
            const size_t batch = 64;
            std::vector<double> row_p(num_classes);
            while (true) {
                size_t start = row_counter.fetch_add(batch);
                if (start >= n_rows) break;
                size_t end = std::min(start + batch, n_rows);
                for (size_t r = start; r < end; ++r) {
                    predict_row_proba(X + r * n_cols, n_cols, row_p);
                    for (size_t c = 0; c < num_classes; ++c) {
                        out_probs_flat[r * num_classes + c] = row_p[c];
                    }
                }
            }
        });
    }

    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }
}

std::unordered_map<std::string, double> RandomForestCPP::feat_importance() const {
    std::unordered_map<std::string, double> imp;
    if (trees.empty()) return imp;
    for (const auto& f : feature_names) imp[f] = 0.0;

    for (const auto& tree : trees) {
        for (int32_t uf : tree.used_features) {
            if (uf >= 0 && uf < static_cast<int32_t>(feature_names.size())) {
                imp[feature_names[uf]] += 1.0;
            }
        }
    }
    double inv_n = 1.0 / static_cast<double>(trees.size());
    for (auto& pair : imp) pair.second *= inv_n;
    return imp;
}

std::unordered_map<std::string, double> RandomForestCPP::feat_score_mean() const {
    std::unordered_map<std::string, double> total_score;
    std::unordered_map<std::string, int32_t> counts;
    for (const auto& f : feature_names) {
        total_score[f] = 0.0;
        counts[f] = 0;
    }

    for (size_t i = 0; i < trees.size(); ++i) {
        double score = survived_scores[i];
        for (int32_t uf : trees[i].used_features) {
            if (uf >= 0 && uf < static_cast<int32_t>(feature_names.size())) {
                const auto& fname = feature_names[uf];
                total_score[fname] += score;
                counts[fname]++;
            }
        }
    }

    std::unordered_map<std::string, double> mean_score;
    for (const auto& f : feature_names) {
        mean_score[f] = (counts[f] > 0) ? (total_score[f] / counts[f]) : 0.0;
    }
    return mean_score;
}

nb::dict RandomForestCPP::feat_pair_importance() const {
    nb::dict result;
    if (trees.empty() || feature_names.size() < 2) return result;

    std::map<std::pair<std::string, std::string>, double> pair_counts;
    double inv_n = 1.0 / static_cast<double>(trees.size());

    for (const auto& tree : trees) {
        std::vector<bool> has_feat(feature_names.size(), false);
        for (int32_t uf : tree.used_features) {
            if (uf >= 0 && uf < static_cast<int32_t>(feature_names.size())) {
                has_feat[uf] = true;
            }
        }
        for (size_t i = 0; i < feature_names.size(); ++i) {
            if (!has_feat[i]) continue;
            for (size_t j = i + 1; j < feature_names.size(); ++j) {
                if (has_feat[j]) {
                    pair_counts[{feature_names[i], feature_names[j]}] += inv_n;
                }
            }
        }
    }

    for (const auto& item : pair_counts) {
        nb::tuple key = nb::make_tuple(item.first.first.c_str(), item.first.second.c_str());
        result[key] = item.second;
    }
    return result;
}

nb::dict RandomForestCPP::to_dict(const std::string& version) const {
    nb::dict root;
    root["min_feature"] = min_feature;
    root["max_feature"] = max_feature;
    root["n_trees"] = n_trees;
    root["target_col"] = target_col.c_str();

    nb::list c_list;
    for (const auto& c : class_names) c_list.append(c.c_str());
    root["class_vals"] = c_list;

    nb::list s_list;
    for (double s : survived_scores) s_list.append(s);
    root["survived_scores"] = s_list;

    root["version"] = version.c_str();

    nb::list num_cols;
    nb::list feat_cols;
    nb::dict type_dict;
    for (size_t i = 0; i < feature_names.size(); ++i) {
        feat_cols.append(feature_names[i].c_str());
        std::string t_str = (feature_types[i] == FeatType::NUMERIC) ? "numeric" : "categorical";
        type_dict[feature_names[i].c_str()] = t_str.c_str();
        if (feature_types[i] == FeatType::NUMERIC) {
            num_cols.append(feature_names[i].c_str());
        }
    }
    root["numeric_cols"] = num_cols;
    root["feature_cols"] = feat_cols;
    root["type_of_cols"] = type_dict;

    nb::list forest_list;
    for (const auto& tree : trees) {
        forest_list.append(tree.to_dict(feature_names, class_names, version));
    }
    root["Forest"] = forest_list;
    return root;
}

void RandomForestCPP::from_dict(nb::dict model_dict) {
    trees.clear();
    survived_scores.clear();

    if (model_dict.contains("min_feature") && !model_dict["min_feature"].is_none()) {
        min_feature = nb::cast<int32_t>(model_dict["min_feature"]);
    }
    if (model_dict.contains("max_feature") && !model_dict["max_feature"].is_none()) {
        max_feature = nb::cast<int32_t>(model_dict["max_feature"]);
    }
    if (model_dict.contains("n_trees")) {
        n_trees = nb::cast<int32_t>(model_dict["n_trees"]);
    }
    if (model_dict.contains("target_col")) {
        target_col = nb::cast<std::string>(model_dict["target_col"]);
    }

    class_names.clear();
    if (model_dict.contains("class_vals")) {
        nb::list c_list = nb::cast<nb::list>(model_dict["class_vals"]);
        for (size_t i = 0; i < c_list.size(); ++i) {
            class_names.push_back(nb::cast<std::string>(c_list[i]));
        }
    }

    feature_names.clear();
    feature_types.clear();
    if (model_dict.contains("feature_cols")) {
        nb::list f_list = nb::cast<nb::list>(model_dict["feature_cols"]);
        nb::dict t_dict;
        if (model_dict.contains("type_of_cols")) {
            t_dict = nb::cast<nb::dict>(model_dict["type_of_cols"]);
        }
        for (size_t i = 0; i < f_list.size(); ++i) {
            std::string fname = nb::cast<std::string>(f_list[i]);
            feature_names.push_back(fname);
            if (t_dict.contains(fname.c_str())) {
                std::string t_str = nb::cast<std::string>(t_dict[fname.c_str()]);
                feature_types.push_back((t_str == "numeric") ? FeatType::NUMERIC : FeatType::CATEGORICAL);
            } else {
                feature_types.push_back(FeatType::NUMERIC);
            }
        }
    }

    if (model_dict.contains("Forest")) {
        nb::list f_list = nb::cast<nb::list>(model_dict["Forest"]);
        trees.reserve(f_list.size());
        survived_scores.reserve(f_list.size());
        for (size_t i = 0; i < f_list.size(); ++i) {
            nb::dict t_dict = nb::cast<nb::dict>(f_list[i]);
            DecisionTreeCPP tree(static_cast<int32_t>(class_names.size()));
            tree.from_dict(t_dict, feature_names, class_names);
            trees.push_back(std::move(tree));
            survived_scores.push_back(trees.back().survived_score);
        }
    }
}

} // namespace rf_mc
