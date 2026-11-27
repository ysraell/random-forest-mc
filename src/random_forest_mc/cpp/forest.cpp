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

SlotResult RandomForestCPP::run_slot(
    const double* X,
    const int32_t* y,
    size_t n_rows,
    size_t n_cols,
    const std::vector<std::vector<int32_t>>& class_indices,
    uint64_t seed,
    int32_t slot
) const {
    std::mt19937_64 rng(seed);
    const int32_t K = static_cast<int32_t>(class_names.size());
    const bool keep_candidates = selection == 1;

    // Class-balanced, disjoint training batch, selection set and scoring set.
    std::vector<int32_t> idx_train, idx_sel, idx_score;
    auto draw_batches = [&]() {
        idx_train.clear();
        idx_sel.clear();
        idx_score.clear();
        for (const auto& c_idx : class_indices) {
            if (c_idx.empty()) continue;
            std::vector<int32_t> pool = c_idx;
            std::shuffle(pool.begin(), pool.end(), rng);
            size_t a = std::min(pool.size(), static_cast<size_t>(std::max(0, n_train_pclass)));
            size_t b = std::min(pool.size(), a + static_cast<size_t>(std::max(0, n_sel_pclass)));
            size_t c = std::min(pool.size(), b + static_cast<size_t>(std::max(0, n_score_pclass)));
            idx_train.insert(idx_train.end(), pool.begin(), pool.begin() + a);
            idx_sel.insert(idx_sel.end(), pool.begin() + a, pool.begin() + b);
            idx_score.insert(idx_score.end(), pool.begin() + b, pool.begin() + c);
        }
    };

    // (accuracy, Brier score); accuracy is 1.0 on an empty set.
    auto score = [&](const DecisionTreeCPP& tree, const std::vector<int32_t>& idx, double& brier) {
        brier = 0.0;
        if (idx.empty()) return 1.0;
        std::vector<double> probs;
        int32_t correct = 0;
        for (int32_t v_idx : idx) {
            tree.predict_row(X + v_idx * n_cols, n_cols, probs, infer_params);
            int32_t best_c = 0;
            for (int32_t c = 1; c < K; ++c) {
                if (probs[c] > probs[best_c]) best_c = c;
            }
            if (best_c == y[v_idx]) correct++;
            for (int32_t c = 0; c < K; ++c) {
                double d = probs[c] - (c == y[v_idx] ? 1.0 : 0.0);
                brier += d * d;
            }
        }
        brier /= static_cast<double>(idx.size());
        return static_cast<double>(correct) / static_cast<double>(idx.size());
    };

    int32_t min_f = (min_feature > 0) ? min_feature : 2;
    int32_t max_f = (max_feature > 0) ? max_feature : static_cast<int32_t>(feature_names.size());
    if (feature_max_sampling > 0) max_f = std::min(max_f, feature_max_sampling);
    min_f = std::min(min_f, static_cast<int32_t>(feature_names.size()));
    max_f = std::min(max_f, static_cast<int32_t>(feature_names.size()));
    if (min_f > max_f) max_f = min_f;

    std::vector<int32_t> all_feats(feature_names.size());
    std::iota(all_feats.begin(), all_feats.end(), 0);

    TreeParams params = tree_params;
    params.n_classes = K;
    params.max_depth = max_depth;
    params.min_samples_split = min_samples_split;

    SlotResult result;
    DecisionTreeCPP best(K), global_best(K);
    bool has_best = false, has_global = false;
    double best_key0 = 0.0, best_key1 = 0.0, global_key0 = 0.0, global_key1 = 0.0;
    int32_t best_rec = -1, global_rec = -1;
    int32_t attempts = 0, batch_id = 0, fails = 0;
    double threshold = th_start;
    DecisionTreeCPP accepted(K);
    int32_t accepted_rec = -1;

    draw_batches();
    while (true) {
        attempts++;
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

        DecisionTreeCPP tree(K);
        tree.plant(X, y, n_rows, n_cols, idx_train, sample_f, feature_types, params);

        double brier = 0.0, unused = 0.0;
        double s_sel = score(tree, idx_sel, brier);
        bool has_score = !idx_score.empty();
        double s_score = has_score ? score(tree, idx_score, unused) : s_sel;
        tree.selection_score = s_sel;
        tree.survived_score = s_score;

        CandidateRecord rec;
        rec.slot = slot;
        rec.batch = batch_id;
        rec.F = tree.feature_sequence;
        rec.used = tree.used_features;
        rec.depth = tree.max_depth_reached;
        rec.s_sel = s_sel;
        rec.s_score = s_score;
        rec.has_score = has_score;
        result.records.push_back(rec);
        int32_t rec_idx = static_cast<int32_t>(result.records.size()) - 1;

        if (keep_candidates) {
            std::vector<double> probs;
            for (int32_t e_idx : ens_sel_idx) {
                tree.predict_row(X + e_idx * n_cols, n_cols, probs, infer_params);
                for (int32_t c = 0; c < K; ++c) result.cand_probs.push_back(static_cast<float>(probs[c]));
            }
            result.cand_trees.push_back(tree);
        }

        double key1 = infer_params.leaf_alpha > 0.0 ? -brier : 0.0;
        if (!has_best || s_sel > best_key0 || (s_sel == best_key0 && key1 > best_key1)) {
            best = tree;
            best_key0 = s_sel;
            best_key1 = key1;
            best_rec = rec_idx;
            has_best = true;
        }
        if (!has_global || s_sel > global_key0 || (s_sel == global_key0 && key1 > global_key1)) {
            global_best = tree;
            global_key0 = s_sel;
            global_key1 = key1;
            global_rec = rec_idx;
            has_global = true;
        }

        if (s_sel >= threshold && s_sel >= tau_min) {
            accepted = std::move(tree);
            accepted_rec = rec_idx;
            break;
        }
        fails++;
        if (max_attempts > 0 && attempts >= max_attempts) {
            accepted = global_best;
            accepted.hit_attempt_cap = true;
            accepted_rec = global_rec;
            break;
        }
        if (fails >= max_discard_trees) {
            if (get_best_tree) {
                if (best_key0 >= tau_min) {
                    accepted = best;
                    accepted_rec = best_rec;
                    break;
                }
                draw_batches();
                batch_id++;
                threshold = th_start;
                fails = 0;
                has_best = false;
            } else if (threshold - delta_th < tau_min) {
                draw_batches();
                batch_id++;
                threshold = th_start;
                fails = 0;
                has_best = false;
            } else {
                threshold -= delta_th;
            }
        }
    }

    accepted.attempts = attempts;
    if (accepted_rec >= 0) result.records[accepted_rec].accepted = true;
    result.tree = std::move(accepted);
    return result;
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
    candidate_log.clear();
    candidate_trees.clear();
    candidate_probs.clear();
    ens_sel_idx.clear();
    ens_score_idx.clear();
    if (n_rows == 0 || n_cols == 0 || class_names.empty()) return;

    // Group sample indices by class
    std::vector<std::vector<int32_t>> class_indices(class_names.size());
    for (size_t i = 0; i < n_rows; ++i) {
        int32_t c = y[i];
        if (c >= 0 && c < static_cast<int32_t>(class_names.size())) {
            class_indices[c].push_back(static_cast<int32_t>(i));
        }
    }

    // Global balanced holdout for ensemble selection, excluded from every slot's batches.
    if (selection == 1) {
        std::mt19937_64 rng(random_seed ^ 0x9E3779B97F4A7C15ULL);
        for (auto& c_idx : class_indices) {
            std::shuffle(c_idx.begin(), c_idx.end(), rng);
            size_t a = std::min(c_idx.size(), static_cast<size_t>(std::max(0, ens_sel_pclass)));
            size_t b = std::min(c_idx.size(), a + static_cast<size_t>(std::max(0, ens_score_pclass)));
            ens_sel_idx.insert(ens_sel_idx.end(), c_idx.begin(), c_idx.begin() + a);
            ens_score_idx.insert(ens_score_idx.end(), c_idx.begin() + a, c_idx.begin() + b);
            c_idx.erase(c_idx.begin(), c_idx.begin() + b);
        }
    }

    int32_t threads_to_use = (n_threads <= 0) ? static_cast<int32_t>(std::thread::hardware_concurrency()) : n_threads;
    if (threads_to_use <= 0) threads_to_use = 1;
    threads_to_use = std::min(threads_to_use, n_trees);

    std::vector<SlotResult> results(n_trees);
    std::atomic<int32_t> next_tree(0);
    std::vector<std::thread> workers;
    workers.reserve(threads_to_use);

    for (int32_t t = 0; t < threads_to_use; ++t) {
        workers.emplace_back([&]() {
            while (true) {
                int32_t tree_idx = next_tree.fetch_add(1);
                if (tree_idx >= n_trees) break;
                // Seed depends only on the slot, so results do not depend on thread scheduling.
                results[tree_idx] = run_slot(
                    X, y, n_rows, n_cols, class_indices, random_seed + static_cast<uint64_t>(tree_idx) * 10007ULL, tree_idx
                );
            }
        });
    }

    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }

    for (auto& res : results) {
        candidate_log.insert(candidate_log.end(), res.records.begin(), res.records.end());
        if (selection == 1) {
            for (auto& t : res.cand_trees) candidate_trees.push_back(std::move(t));
            candidate_probs.insert(candidate_probs.end(), res.cand_probs.begin(), res.cand_probs.end());
        } else {
            trees.push_back(std::move(res.tree));
        }
    }
    survived_scores.resize(trees.size());
    for (size_t i = 0; i < trees.size(); ++i) {
        survived_scores[i] = trees[i].survived_score;
    }
}

void RandomForestCPP::set_trees_from_candidates(const std::vector<int32_t>& chosen) {
    trees.clear();
    survived_scores.clear();
    for (int32_t i : chosen) {
        if (i >= 0 && i < static_cast<int32_t>(candidate_trees.size())) {
            trees.push_back(candidate_trees[i]);
            survived_scores.push_back(candidate_trees[i].survived_score);
        }
    }
}

std::vector<double> RandomForestCPP::tree_weights() const {
    std::vector<double> w(trees.size(), 1.0);
    if (!weighted_tree) return w;
    double total = 0.0;
    double log_k1 = std::log(std::max(1.0, static_cast<double>(class_names.size()) - 1.0));
    for (size_t i = 0; i < trees.size(); ++i) {
        double s = survived_scores[i];
        if (tree_weighting == 1) {
            s = std::min(std::max(s, 1e-3), 1.0 - 1e-3);
            w[i] = std::max(0.0, std::log(s / (1.0 - s)) + log_k1);
        } else {
            w[i] = s;
        }
        total += w[i];
    }
    if (total <= 0.0) std::fill(w.begin(), w.end(), 1.0);
    return w;
}

void RandomForestCPP::predict_row_proba(
    const double* row,
    size_t n_cols,
    std::vector<double>& out_probs
) const {
    size_t num_classes = class_names.size();
    out_probs.assign(num_classes, 0.0);
    if (trees.empty()) return;

    std::vector<double> weights = tree_weights();
    double weight_sum = 0.0;
    for (double w : weights) weight_sum += w;

    std::vector<double> tree_p(num_classes, 0.0);
    for (size_t i = 0; i < trees.size(); ++i) {
        if (soft_voting) {
            trees[i].predict_row(row, n_cols, tree_p, infer_params);
            for (size_t c = 0; c < num_classes; ++c) {
                out_probs[c] += tree_p[c] * weights[i];
            }
        } else {
            int32_t best_c = trees[i].predict_class(row, n_cols, infer_params);
            out_probs[best_c] += weights[i];
        }
    }
    for (size_t c = 0; c < num_classes; ++c) {
        out_probs[c] /= weight_sum;
    }

    // Prior correction P(c|x) ∝ p(c|x)·π_c for forests trained on class-balanced batches.
    if (class_priors.size() == num_classes) {
        double total = 0.0;
        for (size_t c = 0; c < num_classes; ++c) {
            out_probs[c] *= class_priors[c];
            total += out_probs[c];
        }
        if (total > 0.0) {
            for (size_t c = 0; c < num_classes; ++c) out_probs[c] /= total;
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

void RandomForestCPP::trees_proba_batch(
    const double* X,
    size_t n_rows,
    size_t n_cols,
    std::vector<double>& out_flat
) const {
    size_t num_classes = class_names.size();
    out_flat.assign(trees.size() * n_rows * num_classes, 0.0);
    std::vector<double> probs;
    for (size_t t = 0; t < trees.size(); ++t) {
        for (size_t r = 0; r < n_rows; ++r) {
            trees[t].predict_row(X + r * n_cols, n_cols, probs, infer_params);
            for (size_t c = 0; c < num_classes; ++c) {
                out_flat[(t * n_rows + r) * num_classes + c] = probs[c];
            }
        }
    }
}

nb::list RandomForestCPP::candidate_log_list() const {
    nb::list out;
    auto names = [&](const std::vector<int32_t>& idx) {
        nb::list l;
        for (int32_t f : idx) {
            if (f >= 0 && f < static_cast<int32_t>(feature_names.size())) l.append(feature_names[f].c_str());
        }
        return l;
    };
    for (const auto& rec : candidate_log) {
        nb::dict d;
        d["slot"] = rec.slot;
        d["batch"] = rec.batch;
        d["F"] = names(rec.F);
        d["used_features"] = names(rec.used);
        d["depth"] = rec.depth;
        d["s_sel"] = rec.s_sel;
        if (rec.has_score) d["s_score"] = rec.s_score;
        else d["s_score"] = nb::none();
        d["accepted"] = rec.accepted;
        out.append(d);
    }
    return out;
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
