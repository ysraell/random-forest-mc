"""
Forest of trees-based ensemble methods.

Random forests: extremely randomized trees with dynamic tree selection Monte Carlo based.
Modern C++ (C++17 + nanobind) accelerated backend.
"""

import logging as log
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .__init__ import __version__
from .forest import BaseRandomForestMC
from .selection import ensemble_select, ensemble_score
from .model import (
    DatasetNotFound,
    DictValuesAllFeaturesMissing,
    DictValues,
    MissingValuesNotFound,
)
from .tree import (
    DecisionTreeMC,
    LeafDict,
    PandasSeriesRow,
    TypeClassVal,
    featName,
    rowOrMatrix,
)

# Code for categories never seen in training: routed like any non-matching value ('<').
UNKNOWN_CATEGORY_CODE = -1.0

try:
    from ._cpp_forest import DecisionTreeCPP, RandomForestCPP
    CPP_AVAILABLE = True
except ImportError as _err:
    CPP_AVAILABLE = False
    _CPP_IMPORT_ERROR = _err


def _clean_dict(obj):
    if isinstance(obj, dict):
        return {k: _clean_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_dict(v) for v in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    return obj


class RandomForestMC(BaseRandomForestMC):
    """Modern C++ accelerated Random Forest classifier based on Monte Carlo simulations.

    Provides identical API to the pure-Python RandomForestMC with substantial performance
    speedups through native C++17 execution, O(N) median splits, and true GIL-free multi-threading.
    """

    def __init__(
        self,
        n_trees: int = 16,
        target_col: str = "target",
        batch_train_pclass: int = 10,
        batch_val_pclass: int = 10,
        max_discard_trees: int = 10,
        delta_th: float = 0.1,
        th_start: float = 1.0,
        get_best_tree: bool = True,
        min_feature: Optional[int] = None,
        max_feature: Optional[int] = None,
        th_decease_verbose: bool = False,
        temporal_features: bool = False,
        split_with_replace: bool = False,
        max_depth: Optional[int] = None,
        min_samples_split: int = 1,
        got_best_tree_verbose: bool = False,
        threaded_fit: bool = True,
        n_threads: int = 0,
        random_seed: int = 42,
        **v160_params,
    ) -> None:
        if not CPP_AVAILABLE:
            raise ImportError(
                f"C++ extension (_cpp_forest) is not available: {_CPP_IMPORT_ERROR}. "
                "Ensure build dependencies are installed and run 'python build.py build_ext --inplace'."
            )

        super().__init__(
            n_trees=n_trees,
            target_col=target_col,
            min_feature=min_feature,
            max_feature=max_feature,
            temporal_features=temporal_features,
        )

        if split_with_replace:
            log.warning("'split_with_replace' is deprecated and has no effect.")
        self.split_with_replace = split_with_replace
        if th_decease_verbose:
            log.basicConfig(level=log.INFO)
        self.batch_train_pclass = batch_train_pclass
        self.batch_val_pclass = batch_val_pclass
        self._N = batch_train_pclass + batch_val_pclass
        self.th_start = th_start
        self.delta_th = delta_th
        self.max_discard_trees = max_discard_trees
        self.max_depth = 1000 if max_depth is None else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.got_best_tree_verbose = got_best_tree_verbose
        self.get_best_tree = get_best_tree
        self.threaded_fit = threaded_fit
        self.n_threads = n_threads
        self.random_seed = random_seed
        self._init_v160_params(**v160_params)

        self.dataset = None
        self.dataset_numpy = None
        self.cat_to_code: Dict[str, Dict[Any, float]] = {}
        self.code_to_cat: Dict[str, Dict[float, Any]] = {}
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

        self._cpp_engine = RandomForestCPP()
        self._sync_params_to_cpp()

    def _sync_params_to_cpp(self) -> None:
        self._cpp_engine.n_trees = self.n_trees
        self._cpp_engine.target_col = self.target_col
        self._cpp_engine.batch_train_pclass = self.batch_train_pclass
        self._cpp_engine.batch_val_pclass = self.batch_val_pclass
        self._cpp_engine.max_discard_trees = self.max_discard_trees
        self._cpp_engine.delta_th = self.delta_th
        self._cpp_engine.th_start = self.th_start
        self._cpp_engine.get_best_tree = self.get_best_tree
        self._cpp_engine.min_feature = -1 if self.min_feature is None else self.min_feature
        self._cpp_engine.max_feature = -1 if self.max_feature is None else self.max_feature
        self._cpp_engine.temporal_features = self.temporal_features
        self._cpp_engine.max_depth = self.max_depth
        self._cpp_engine.min_samples_split = self.min_samples_split
        self._cpp_engine.soft_voting = self.soft_voting
        self._cpp_engine.weighted_tree = self.weighted_tree
        self._cpp_engine.random_seed = self.random_seed
        self._cpp_engine.set_tree_params(
            0 if self.split_rule == "legacy" else 1,
            self.split_supervision,
            self.split_lookahead,
            0 if self.categorical_split == "mode" else 1,
        )
        self._cpp_engine.set_infer_params(0 if self.missing_strategy == "weighted" else 1, self.leaf_alpha)
        self._cpp_engine.tree_weighting = 1 if self.tree_weighting == "samme" else 0
        self._cpp_engine.selection = 1 if self.selection == "ensemble" else 0
        priors = self._resolved_priors() if self.class_vals else None
        self._cpp_engine.class_priors = (
            [float(priors.get(str(c), priors.get(c, 0.0))) for c in self.class_vals] if priors else []
        )
        tau_min = self._tau_min() if self.class_vals else -np.inf
        self._cpp_engine.tau_min = float(tau_min) if np.isfinite(tau_min) else -1e300
        max_attempts = self._max_attempts()
        self._cpp_engine.max_attempts = 0 if max_attempts is None else int(max_attempts)
        if getattr(self, "_min_class", None):
            ens_pool = self._ensemble_pool_size(self._min_class)
            ens_score = int(self.score_split * ens_pool)
            if ens_score >= ens_pool:
                ens_score = ens_pool - 1
            self._cpp_engine.ens_sel_pclass = ens_pool - ens_score
            self._cpp_engine.ens_score_pclass = ens_score
            n_train, n_sel, n_score = self._batch_sizes(self._min_class - ens_pool)
            self._cpp_engine.n_train_pclass = n_train
            self._cpp_engine.n_sel_pclass = n_sel
            self._cpp_engine.n_score_pclass = n_score
            self.L_eff = self._effective_depth(n_train)
            if self.feature_sampling == "effective_depth":
                self._cpp_engine.feature_max_sampling = max(self.min_feature or 2, self.L_eff)
            else:
                self._cpp_engine.feature_max_sampling = -1

    def process_dataset(self, dataset: pd.DataFrame) -> None:
        dataset = dataset.copy()
        dataset[self.target_col] = dataset[self.target_col].astype(str)
        feature_cols = [col for col in dataset.columns if col != self.target_col]
        numeric_cols = dataset.select_dtypes([np.number]).columns.to_list()
        categorical_cols = list(set(feature_cols) - set(numeric_cols))
        type_of_cols = {col: "numeric" for col in numeric_cols}
        type_of_cols.update({col: "categorical" for col in categorical_cols})

        if self.min_feature is None:
            self.min_feature = 2

        if self.max_feature is None:
            self.max_feature = len(feature_cols)

        self.numeric_cols = numeric_cols
        self.feature_cols = feature_cols
        self.type_of_cols = type_of_cols

        dataset = dataset.dropna()
        log.warning("Rows with missing values were dropped from the dataset.")

        self.class_vals = sorted(dataset[self.target_col].unique().tolist())
        self._set_class_freqs(dataset[self.target_col])
        self.class_to_idx = {c: i for i, c in enumerate(self.class_vals)}
        self.idx_to_class = {i: c for i, c in enumerate(self.class_vals)}

        if self.temporal_features and (not self.validFeaturesTemporal()):
            self.temporal_features = False
            log.warning("Temporal features ordering disabled: missing orderable features!")

        min_class = dataset[self.target_col].value_counts().min()
        self._N = min(self._N, min_class)
        self._min_class = int(min_class)
        self.dataset_numpy = {col: dataset[col].to_numpy() for col in dataset.columns}
        self.n_samples = len(dataset)

        # Build categorical encodings
        self.cat_to_code.clear()
        self.code_to_cat.clear()
        for col in categorical_cols:
            cats = dataset[col].unique()
            c2code = {val: float(idx) for idx, val in enumerate(cats)}
            code2c = {float(idx): val for idx, val in enumerate(cats)}
            self.cat_to_code[col] = c2code
            self.code_to_cat[col] = code2c

        # Build contiguous float64 matrix X and int32 target vector y
        X_mat = np.empty((len(dataset), len(feature_cols)), dtype=np.float64, order="C")
        for j, col in enumerate(feature_cols):
            if col in numeric_cols:
                X_mat[:, j] = dataset[col].to_numpy(dtype=np.float64)
            else:
                c2code = self.cat_to_code[col]
                X_mat[:, j] = dataset[col].map(c2code).to_numpy(dtype=np.float64)

        y_vec = dataset[self.target_col].map(self.class_to_idx).to_numpy(dtype=np.int32)

        self._X = X_mat
        self._y = y_vec

        feat_types = [type_of_cols[col] for col in feature_cols]
        self._cpp_engine.set_features_and_classes(feature_cols, self.class_vals, feat_types)
        self._sync_params_to_cpp()

    def fit(self, dataset: Optional[pd.DataFrame] = None, disable_progress_bar: bool = False) -> None:
        if dataset is not None:
            self.process_dataset(dataset)

        if self.dataset_numpy is None:
            raise DatasetNotFound

        self._sync_params_to_cpp()
        self._cpp_engine.fit(self._X, self._y, self.n_threads)
        self.candidate_log = list(self._cpp_engine.candidate_log_list())

        if self.selection == "ensemble":
            K = len(self.class_vals)
            ens_sel = np.asarray(self._cpp_engine.ens_sel_idx, dtype=np.int64)
            ens_score_idx = np.asarray(self._cpp_engine.ens_score_idx, dtype=np.int64)
            n_cand = self._cpp_engine.candidate_count()
            probs = np.asarray(self._cpp_engine.candidate_probs(), dtype=np.float32).reshape(n_cand, len(ens_sel), K)
            chosen = ensemble_select(
                probs, self._y[ens_sel], n_select=self.n_trees, n_init=self.ensemble_init, metric=self.ensemble_metric
            )
            self._cpp_engine.set_trees_from_candidates(chosen)
            self._cpp_engine.clear_candidates()
            eval_idx = ens_score_idx if len(ens_score_idx) else ens_sel
            X_eval = np.ascontiguousarray(self._X[eval_idx])
            tree_probs = np.asarray(self._cpp_engine.trees_proba_batch(X_eval)).reshape(len(chosen), len(eval_idx), K)
            self.ensemble_score = ensemble_score(tree_probs, self._y[eval_idx])

        self.survived_scores = list(self._cpp_engine.survived_scores)
        # Populate self.data with DecisionTreeMC wrappers for BaseRandomForestMC compatibility
        self.data = [DecisionTreeMC.from_dict(Tree) for Tree in self._engine_dict()["Forest"]]
        self._warn_attempt_cap()

    def fitParallel(
        self,
        dataset: Optional[pd.DataFrame] = None,
        disable_progress_bar: bool = False,
        max_workers: Optional[int] = None,
    ) -> None:
        saved_threads = self.n_threads
        if max_workers is not None:
            self.n_threads = max_workers
        try:
            self.fit(dataset=dataset, disable_progress_bar=disable_progress_bar)
        finally:
            self.n_threads = saved_threads

    def _row_to_c_array(self, row: Union[PandasSeriesRow, Dict[str, Any]]) -> np.ndarray:
        arr = np.empty(len(self.feature_cols), dtype=np.float64)
        for j, col in enumerate(self.feature_cols):
            val = row.get(col) if isinstance(row, dict) else (row[col] if col in row else np.nan)
            if pd.isna(val):
                arr[j] = np.nan
            elif col in self.numeric_cols:
                arr[j] = float(val)
            else:
                c2code = self.cat_to_code.get(col, {})
                arr[j] = c2code.get(val, UNKNOWN_CATEGORY_CODE)
        return arr

    def _df_to_c_matrix(self, df: pd.DataFrame) -> np.ndarray:
        mat = np.empty((len(df), len(self.feature_cols)), dtype=np.float64, order="C")
        for j, col in enumerate(self.feature_cols):
            if col not in df.columns:
                mat[:, j] = np.nan
            elif col in self.numeric_cols:
                mat[:, j] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            else:
                c2code = self.cat_to_code.get(col, {})
                codes = df[col].map(c2code).to_numpy(dtype=np.float64, copy=True)
                unknown = np.isnan(codes) & ~df[col].isna().to_numpy()
                codes[unknown] = UNKNOWN_CATEGORY_CODE
                mat[:, j] = codes
        return mat

    def useForest(self, row: PandasSeriesRow) -> LeafDict:
        self._sync_params_to_cpp()
        row_arr = self._row_to_c_array(row)
        probs = self._cpp_engine.predict_row_proba(row_arr)
        return {c: probs[i] for i, c in enumerate(self.class_vals)}

    def predict_proba(self, row_or_matrix: rowOrMatrix, prob_output: bool = True) -> Union[LeafDict, List[LeafDict]]:
        self._sync_params_to_cpp()
        if isinstance(row_or_matrix, (PandasSeriesRow, dict)):
            return self.useForest(row_or_matrix)
        if isinstance(row_or_matrix, pd.DataFrame):
            mat = self._df_to_c_matrix(row_or_matrix)
            flat_probs = self._cpp_engine.predict_proba_batch(mat, self.n_threads)
            n_classes = len(self.class_vals)
            result: List[LeafDict] = []
            for r in range(len(row_or_matrix)):
                row_dict = {
                    self.class_vals[c]: flat_probs[r * n_classes + c]
                    for c in range(n_classes)
                }
                result.append(row_dict)
            return result
        raise TypeError("The input argument must be a Pandas Series, Dict, or Pandas DataFrame.")

    def predict(
        self, row_or_matrix: rowOrMatrix, prob_output: bool = False
    ) -> Union[LeafDict, List[TypeClassVal], List[LeafDict]]:
        if prob_output:
            return self.predict_proba(row_or_matrix, prob_output=True)
        if isinstance(row_or_matrix, (PandasSeriesRow, dict)):
            probs = self.useForest(row_or_matrix)
            return self._decide(probs)
        if isinstance(row_or_matrix, pd.DataFrame):
            if self.decision_threshold is not None:
                return [self._decide(dist) for dist in self.predict_proba(row_or_matrix)]
            mat = self._df_to_c_matrix(row_or_matrix)
            class_indices = self._cpp_engine.predict_batch(mat, self.n_threads)
            return [self.idx_to_class[idx] for idx in class_indices]
        raise TypeError("The input argument must be a Pandas Series, Dict, or Pandas DataFrame.")

    def testForest(self, ds: pd.DataFrame) -> List[TypeClassVal]:
        return self.predict(ds, prob_output=False)

    def testForestParallel(
        self,
        ds: pd.DataFrame,
        max_workers: Optional[int] = None,
        chunksize: Optional[int] = None,
    ) -> List[TypeClassVal]:
        saved_threads = self.n_threads
        if max_workers is not None:
            self.n_threads = max_workers
        try:
            return self.predict(ds, prob_output=False)
        finally:
            self.n_threads = saved_threads

    def testForestProbs(self, ds: pd.DataFrame) -> List[LeafDict]:
        return self.predict_proba(ds, prob_output=True)

    def testForestProbsParallel(
        self,
        ds: pd.DataFrame,
        max_workers: Optional[int] = None,
        chunksize: Optional[int] = None,
    ) -> List[LeafDict]:
        saved_threads = self.n_threads
        if max_workers is not None:
            self.n_threads = max_workers
        try:
            return self.predict_proba(ds, prob_output=True)
        finally:
            self.n_threads = saved_threads

    def featImportance(self, Forest: Optional[List[DecisionTreeMC]] = None) -> Dict[featName, float]:
        if Forest is None:
            return self._cpp_engine.feat_importance()
        return super().featImportance(Forest=Forest)

    def featScoreMean(self, Forest: Optional[List[DecisionTreeMC]] = None) -> Dict[featName, float]:
        if Forest is None:
            return self._cpp_engine.feat_score_mean()
        return super().featScoreMean(Forest=Forest)

    def featPairImportance(
        self, disable_progress_bar: bool = False, Forest: Optional[List[DecisionTreeMC]] = None
    ) -> Dict[Tuple[featName, featName], float]:
        if Forest is None:
            return self._cpp_engine.feat_pair_importance()
        return super().featPairImportance(disable_progress_bar=disable_progress_bar, Forest=Forest)

    def featCorrDataFrame(self, Forest: Optional[List[DecisionTreeMC]] = None) -> pd.DataFrame:
        N = len(self.feature_cols)
        matrix = np.zeros((N, N), dtype=np.float16)
        for feat, count in self.featImportance(Forest=Forest).items():
            if feat in self.feature_cols:
                idx = self.feature_cols.index(feat)
                matrix[idx][idx] = count

        for pair, count in self.featPairImportance(Forest=Forest).items():
            if pair[0] in self.feature_cols and pair[1] in self.feature_cols:
                idxa = self.feature_cols.index(pair[0])
                idxb = self.feature_cols.index(pair[1])
                matrix[idxa][idxb], matrix[idxb][idxa] = count, count

        return pd.DataFrame(matrix, index=self.feature_cols, columns=self.feature_cols)

    def sampleClass2trees(self, row: PandasSeriesRow, Class: TypeClassVal) -> List[DecisionTreeMC]:
        return [Tree for Tree in self.data if self.maxProbClas(Tree(row)) == Class]

    def sampleClassFeatImportance(self, row: PandasSeriesRow, Class: TypeClassVal) -> Dict[featName, float]:
        return self.featImportance(self.sampleClass2trees(row=row, Class=Class))

    def sampleClassFeatScoreMean(self, row: PandasSeriesRow, Class: TypeClassVal) -> Dict[featName, float]:
        return self.featScoreMean(self.sampleClass2trees(row=row, Class=Class))

    def sampleClassFeatPairImportance(
        self, row: PandasSeriesRow, Class: TypeClassVal
    ) -> Dict[Tuple[featName, featName], float]:
        return self.featPairImportance(Forest=self.sampleClass2trees(row=row, Class=Class))

    def sampleClassFeatCorrDataFrame(self, row: PandasSeriesRow, Class: TypeClassVal) -> pd.DataFrame:
        return self.featCorrDataFrame(self.sampleClass2trees(row=row, Class=Class))

    def featCount(
        self, Forest: Optional[List[DecisionTreeMC]] = None
    ) -> Tuple[Tuple[float, float, int, int], List[int]]:
        if Forest is None:
            Forest = self.data
        out = [len(Tree.used_features) for Tree in Forest]
        return (np.mean(out), np.std(out), min(out), max(out)), out

    def sampleClassFeatCount(
        self, row: PandasSeriesRow, Class: TypeClassVal
    ) -> Tuple[Tuple[float, float, int, int], List[int]]:
        return self.featCount(self.sampleClass2trees(row=row, Class=Class))

    # --- Serialization (category values in the dict, codes inside the C++ engine) ---

    def _map_split_values(self, node: dict, mapping: Dict[str, Dict[Any, Any]]) -> dict:
        """Copy of a tree-dict with categorical split values translated through `mapping`."""
        if "leaf" in node:
            return dict(node)
        feat = next(iter(node.keys()))
        split = dict(node[feat]["split"])
        if split["feat_type"] in ("categorical", "categorical_set") and feat in mapping:
            table = mapping[feat]
            if split["feat_type"] == "categorical_set":
                split["split_val"] = [table.get(v, v) for v in split["split_val"]]
            else:
                split["split_val"] = table.get(split["split_val"], split["split_val"])
        split[">="] = self._map_split_values(split[">="], mapping)
        split["<"] = self._map_split_values(split["<"], mapping)
        return {feat: {"split": split}}

    def _engine_dict(self) -> dict:
        """Engine dict with category values (instead of codes) in the categorical splits."""
        out = _clean_dict(self._cpp_engine.to_dict(__version__))
        for tree in out.get("Forest", []):
            if tree.get("data"):
                tree["data"] = self._map_split_values(tree["data"], self.code_to_cat)
        return out

    def _register_categories(self, node: dict) -> None:
        """Adds codes for categorical split values not yet in `cat_to_code` (models loaded from a dict)."""
        if "leaf" in node:
            return
        feat = next(iter(node.keys()))
        split = node[feat]["split"]
        if split["feat_type"] in ("categorical", "categorical_set"):
            values = split["split_val"] if split["feat_type"] == "categorical_set" else [split["split_val"]]
            c2code = self.cat_to_code.setdefault(feat, {})
            code2c = self.code_to_cat.setdefault(feat, {})
            for val in values:
                if val not in c2code:
                    code = float(len(c2code))
                    c2code[val] = code
                    code2c[code] = val
        self._register_categories(split[">="])
        self._register_categories(split["<"])

    def model2dict(self, include_candidates: bool = False) -> dict:
        out = self._engine_dict()
        out.update({attr: getattr(self, attr, None) for attr in self.attr_to_save_optional})
        if include_candidates:
            out["candidate_log"] = self.candidate_log
        return out

    def dict2model(self, dict_model: dict, add: bool = False) -> None:
        dict_model = _clean_dict(dict_model)
        new_trees = dict_model.get("Forest", [])
        if add:
            new_trees = self.model2dict()["Forest"] + new_trees
        for tree in new_trees:
            if tree.get("data"):
                self._register_categories(tree["data"])
        engine_dict = dict(dict_model)
        engine_dict["Forest"] = [
            dict(tree, data=self._map_split_values(tree["data"], self.cat_to_code) if tree.get("data") else tree.get("data"))
            for tree in new_trees
        ]
        self._cpp_engine.from_dict(engine_dict)
        for attr in self.attr_to_save:
            if attr in dict_model and attr not in ("Forest", "survived_scores"):
                setattr(self, attr, dict_model[attr])
        for attr in self.attr_to_save_optional:
            if attr in dict_model:
                setattr(self, attr, dict_model[attr])
        self.tree_format = dict_model.get("tree_format", 1)
        if "candidate_log" in dict_model:
            self.candidate_log = dict_model["candidate_log"]
        self.class_vals = dict_model.get("class_vals", [])
        self.class_to_idx = {c: i for i, c in enumerate(self.class_vals)}
        self.idx_to_class = {i: c for i, c in enumerate(self.class_vals)}
        self.feature_cols = dict_model.get("feature_cols", [])
        self.numeric_cols = dict_model.get("numeric_cols", [])
        self.survived_scores = list(self._cpp_engine.survived_scores)
        self.data = [DecisionTreeMC.from_dict(Tree) for Tree in new_trees]
        self._sync_params_to_cpp()

    @staticmethod
    def _fill_row_missing(row: PandasSeriesRow, dict_values: DictValues) -> pd.DataFrame:
        list_out = []
        for col, vals in dict_values.items():
            if pd.isna(row[col]):
                for val in vals:
                    _row = row.astype(object).copy()
                    _row[col] = val
                    list_out.append(_row)
        if len(list_out) == 0:
            log.warning("Filling rows process: found row without missing data!")
            return None
        return pd.concat(list_out, axis=1).transpose().reset_index(drop=True)

    def _validationMissingValues(self, dict_values: DictValues) -> None:
        used_features = set()
        for Tree in self:
            used_features |= set(Tree.used_features)
        not_have_feats = set(dict_values.keys()) - used_features
        if not_have_feats:
            _tmp = ", ".join(not_have_feats)
            log.warning(f"The Forest model does not have the following feature(s): [{_tmp}].")
        if len(set(dict_values.keys())) == len(not_have_feats):
            raise DictValuesAllFeaturesMissing

    def _genFilledDataMissing(
        self, row_or_matrix: rowOrMatrix, dict_values: DictValues
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(row_or_matrix, PandasSeriesRow):
            df_data_miss = self._fill_row_missing(row_or_matrix, dict_values)
            if df_data_miss is None:
                raise MissingValuesNotFound
            row_or_matrix = pd.DataFrame(row_or_matrix).transpose().reset_index(drop=True)
        elif isinstance(row_or_matrix, pd.DataFrame):
            row_or_matrix = row_or_matrix.reset_index(drop=True)
            df_data_miss = []
            for _, row in row_or_matrix.iterrows():
                _tmp = self._fill_row_missing(row, dict_values)
                if _tmp is not None:
                    df_data_miss.append(_tmp)
            if len(df_data_miss) == 0:
                raise MissingValuesNotFound
            df_data_miss = pd.concat(df_data_miss).reset_index(drop=True)
        return row_or_matrix, df_data_miss

    def predictMissingValues(self, row_or_matrix: rowOrMatrix, dict_values: DictValues):
        self._validationMissingValues(dict_values)
        row_or_matrix, df_data_miss = self._genFilledDataMissing(row_or_matrix, dict_values)
        df_predict = pd.DataFrame.from_dict(self.predict_proba(df_data_miss))
        df_predict = pd.concat([df_data_miss, df_predict], axis=1)

        out = []
        for i, row in row_or_matrix.reset_index(drop=True).iterrows():
            cols = list(dict_values.keys())
            cond = df_data_miss[cols[0]] == row[cols[0]]
            for col in cols[1:]:
                if not pd.isna(row[col]):
                    cond = cond & (df_data_miss[col] == row[col])
            df_tmp = df_predict.loc[cond]
            df_tmp = pd.concat([pd.DataFrame(row).transpose(), df_tmp]).drop_duplicates().reset_index(drop=True)
            df_tmp["row_id"] = i
            out.append(df_tmp)

        return pd.concat(out).reset_index(drop=True)
