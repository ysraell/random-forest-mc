#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include "forest.hpp"
#include "tree.hpp"

namespace nb = nanobind;
using namespace rf_mc;

NB_MODULE(_cpp_forest, m) {
    m.doc() = "C++17 optimized core engine for random-forest-mc";

    nb::class_<DecisionTreeCPP>(m, "DecisionTreeCPP")
        .def(nb::init<>())
        .def(nb::init<int32_t>())
        .def_rw("survived_score", &DecisionTreeCPP::survived_score)
        .def_ro("used_features", &DecisionTreeCPP::used_features)
        .def("to_dict", &DecisionTreeCPP::to_dict, nb::arg("feature_names"), nb::arg("class_names"), nb::arg("module_version"))
        .def("from_dict", &DecisionTreeCPP::from_dict, nb::arg("tree_dict"), nb::arg("feature_names"), nb::arg("class_names"));

    nb::class_<RandomForestCPP>(m, "RandomForestCPP")
        .def(nb::init<>())
        .def_rw("n_trees", &RandomForestCPP::n_trees)
        .def_rw("target_col", &RandomForestCPP::target_col)
        .def_rw("batch_train_pclass", &RandomForestCPP::batch_train_pclass)
        .def_rw("batch_val_pclass", &RandomForestCPP::batch_val_pclass)
        .def_rw("max_discard_trees", &RandomForestCPP::max_discard_trees)
        .def_rw("delta_th", &RandomForestCPP::delta_th)
        .def_rw("th_start", &RandomForestCPP::th_start)
        .def_rw("get_best_tree", &RandomForestCPP::get_best_tree)
        .def_rw("min_feature", &RandomForestCPP::min_feature)
        .def_rw("max_feature", &RandomForestCPP::max_feature)
        .def_rw("temporal_features", &RandomForestCPP::temporal_features)
        .def_rw("max_depth", &RandomForestCPP::max_depth)
        .def_rw("min_samples_split", &RandomForestCPP::min_samples_split)
        .def_rw("soft_voting", &RandomForestCPP::soft_voting)
        .def_rw("weighted_tree", &RandomForestCPP::weighted_tree)
        .def_rw("random_seed", &RandomForestCPP::random_seed)
        .def_ro("survived_scores", &RandomForestCPP::survived_scores)
        .def_ro("feature_names", &RandomForestCPP::feature_names)
        .def_ro("class_names", &RandomForestCPP::class_names)
        .def("set_features_and_classes", &RandomForestCPP::set_features_and_classes,
             nb::arg("feature_names"), nb::arg("class_names"), nb::arg("feature_types"))
        .def("fit", [](RandomForestCPP& self,
                       nb::ndarray<const double, nb::c_contig> X,
                       nb::ndarray<const int32_t, nb::c_contig> y,
                       int32_t n_threads) {
            size_t n_rows = X.shape(0);
            size_t n_cols = X.shape(1);
            nb::gil_scoped_release release;
            self.fit(X.data(), y.data(), n_rows, n_cols, n_threads);
        }, nb::arg("X"), nb::arg("y"), nb::arg("n_threads") = 0)
        .def("predict_row_proba", [](const RandomForestCPP& self, nb::ndarray<const double, nb::c_contig> row) {
            std::vector<double> probs;
            self.predict_row_proba(row.data(), row.shape(0), probs);
            return probs;
        }, nb::arg("row"))
        .def("predict_batch", [](const RandomForestCPP& self,
                                nb::ndarray<const double, nb::c_contig> X,
                                int32_t n_threads) {
            size_t n_rows = X.shape(0);
            size_t n_cols = X.shape(1);
            std::vector<int32_t> out_classes;
            {
                nb::gil_scoped_release release;
                self.predict_batch(X.data(), n_rows, n_cols, out_classes, n_threads);
            }
            return out_classes;
        }, nb::arg("X"), nb::arg("n_threads") = 0)
        .def("predict_proba_batch", [](const RandomForestCPP& self,
                                      nb::ndarray<const double, nb::c_contig> X,
                                      int32_t n_threads) {
            size_t n_rows = X.shape(0);
            size_t n_cols = X.shape(1);
            std::vector<double> out_probs;
            {
                nb::gil_scoped_release release;
                self.predict_proba_batch(X.data(), n_rows, n_cols, out_probs, n_threads);
            }
            return out_probs;
        }, nb::arg("X"), nb::arg("n_threads") = 0)
        .def("feat_importance", &RandomForestCPP::feat_importance)
        .def("feat_score_mean", &RandomForestCPP::feat_score_mean)
        .def("feat_pair_importance", &RandomForestCPP::feat_pair_importance)
        .def("to_dict", &RandomForestCPP::to_dict, nb::arg("version"))
        .def("from_dict", &RandomForestCPP::from_dict, nb::arg("model_dict"))
        .def("trees_count", [](const RandomForestCPP& self) {
            return self.trees.size();
        });
}
