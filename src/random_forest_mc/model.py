"""
Forest of trees-based ensemble methods.

Random forests: extremely randomized trees with dynamic tree selection Monte Carlo based.

"""

import logging as log
import re
from collections import defaultdict
from itertools import combinations
from numbers import Real
from random import sample as random_sample
from typing import Any, List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Dict
from sys import getrecursionlimit
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from .forest import BaseRandomForestMC
from .selection import ensemble_select, ensemble_score
from .tree import (
    DecisionTreeMC,
    TypeClassVal,
    LeafDict,
    rowOrMatrix,
    featName,
    PandasSeriesRow,
    route_matrix,
)


# For extract the feature names from the tree-dict.
re_feat_name = re.compile("'[\\w\\s]+'\\:")

DictValues = Dict[featName, Union[str, Real]]


def _py_scalar(val: Any) -> Any:
    """Plain Python scalar for JSON-friendly split values."""
    return val.item() if hasattr(val, "item") else val


# Custom exception when missing values not found
class MissingValuesNotFound(Exception):
    """Exception raised for missing values not found.

    Attributes:
        message -- explanation of the error
    """

    def __init__(
        self,
        message="Dataset or row without missing values! Please, give a dataset or row with missing values using 'NaN'.",
    ):
        super().__init__(message)


# Custom exception when no dataset given
class DatasetNotFound(Exception):
    """Exception raised for dataset not found.

    Attributes:
        message -- explanation of the error
    """

    def __init__(
        self,
        message="Dataset not found! Please, give a dataset for functions fit() or process_dataset().",
    ):
        super().__init__(message)


# Custom exception when no dataset given
class DictValuesAllFeaturesMissing(Exception):
    """Exception raised when all features in 'dict_values' are not found in the trained model.

    Attributes:
        message -- explanation of the error
    """

    def __init__(
        self,
        message="All features in the given dictionary 'dict_values' are not found int he trained model (forest).",
    ):
        super().__init__(message)


class RandomForestMC(BaseRandomForestMC):
    """A Random Forest classifier based on Monte Carlo simulations.

    This class extends BaseRandomForestMC to provide a complete random forest
    implementation, including training, prediction, and feature importance analysis.
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
        threaded_fit: bool = False,
        **v160_params,
    ) -> None:
        """Initializes the RandomForestMC object.

        Args:
            n_trees (int, optional): The number of trees in the forest. Defaults to 16.
            target_col (str, optional): The name of the target column. Defaults to "target".
            batch_train_pclass (int, optional): Number of samples per class for training. Defaults to 10.
            batch_val_pclass (int, optional): Number of samples per class for validation. Defaults to 10.
            max_discard_trees (int, optional): Maximum number of trees to discard during training. Defaults to 10.
            delta_th (float, optional): Threshold decrease for tree validation. Defaults to 0.1.
            th_start (float, optional): Starting threshold for tree validation. Defaults to 1.0.
            get_best_tree (bool, optional): Whether to keep the best tree if threshold is not met. Defaults to True.
            min_feature (Optional[int], optional): Minimum number of features to consider for each tree. Defaults to None.
            max_feature (Optional[int], optional): Maximum number of features to consider for each tree. Defaults to None.
            th_decease_verbose (bool, optional): Whether to log threshold decrease. Defaults to False.
            temporal_features (bool, optional): Whether to consider temporal features. Defaults to False.
            split_with_replace (bool, optional): Deprecated and ignored. Defaults to False.
            max_depth (Optional[int], optional): Maximum depth of the trees. Defaults to None (no limit).
            min_samples_split (int, optional): Minimum number of samples required to split an internal node. Defaults to 1.
            got_best_tree_verbose (bool, optional): Whether to log when the best tree is found. Defaults to False.
            **v160_params: Hyperparameters added in v1.6.0 (see `BaseRandomForestMC._init_v160_params`).
        """

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
        self.dataset = None
        self.dataset_numpy = None
        self.max_depth = getrecursionlimit() if max_depth is None else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.got_best_tree_verbose = got_best_tree_verbose
        self.get_best_tree = get_best_tree
        self.threaded_fit = threaded_fit
        self._init_v160_params(**v160_params)

    def process_dataset(self, dataset: pd.DataFrame) -> None:
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

        # Drop missing values
        dataset = dataset.dropna()
        log.warning("Rows with missing values were dropped from the dataset.")

        self.class_vals = dataset[self.target_col].unique().tolist()
        self._set_class_freqs(dataset[self.target_col])

        if self.temporal_features and (not self.validFeaturesTemporal()):
            self.temporal_features = False
            log.warning("Temporal features ordering disable: you do not have all orderable features!")

        min_class = dataset[self.target_col].value_counts().min()
        self._N = min(self._N, min_class)

        # Convert to numpy arrays
        self.dataset_numpy = {col: dataset[col].to_numpy() for col in dataset.columns}
        self.n_samples = len(dataset)
        class_to_code = {c: i for i, c in enumerate(self.class_vals)}
        self._y_codes = np.array([class_to_code[c] for c in self.dataset_numpy[self.target_col]], dtype=np.int64)
        indices = np.arange(self.n_samples)
        self._class_indices = [indices[self._y_codes == i] for i in range(len(self.class_vals))]
        self.L_eff = self._effective_depth(self._batch_sizes(int(min_class))[0])

    # --- Batches ---

    def _draw_batches(self, class_indices: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Class-balanced, disjoint training batch, selection set and scoring set."""
        min_class = min(len(idx) for idx in class_indices)
        n_train, n_sel, n_score = self._batch_sizes(min_class)
        idx_train, idx_sel, idx_score = [], [], []
        for idx in class_indices:
            chosen = np.random.choice(idx, n_train + n_sel + n_score, replace=False)
            idx_train.extend(chosen[:n_train])
            idx_sel.extend(chosen[n_train : n_train + n_sel])
            idx_score.extend(chosen[n_train + n_sel :])
        return np.array(idx_train, dtype=int), np.array(idx_sel, dtype=int), np.array(idx_score, dtype=int)

    # Splits the data to build the decision tree.
    def split_train_val(self) -> Tuple[np.ndarray, np.ndarray]:
        idx_train, idx_sel, idx_score = self._draw_batches(self._class_indices)
        return idx_train, np.concatenate([idx_sel, idx_score]).astype(int)

    def _carve_ensemble_pool(self) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """Global balanced holdout for ensemble selection, excluded from every slot's batches."""
        min_class = min(len(idx) for idx in self._class_indices)
        n_pool = self._ensemble_pool_size(min_class)
        n_score = int(self.score_split * n_pool)
        if n_score >= n_pool:
            n_score = n_pool - 1
        remaining, ens_sel, ens_score = [], [], []
        for idx in self._class_indices:
            perm = np.random.permutation(idx)
            ens_sel.extend(perm[: n_pool - n_score])
            ens_score.extend(perm[n_pool - n_score : n_pool])
            remaining.append(perm[n_pool:])
        return remaining, np.array(ens_sel, dtype=int), np.array(ens_score, dtype=int)

    # Sample the features.
    def sampleFeats(self, feature_cols: List[str]) -> List[featName]:
        # feature_cols already excludes target_col
        max_f = self.max_feature
        if self.feature_sampling == "effective_depth" and self.L_eff is not None:
            max_f = max(self.min_feature, min(self.max_feature, self.L_eff))
        n_samples = np.random.randint(self.min_feature, max_f + 1)
        out = random_sample(feature_cols, min(len(feature_cols), n_samples))
        if not self.temporal_features:
            return out
        return sorted(out, key=lambda x: int(x.split("_")[-1]))

    # --- Tree growing ---

    # Set the leaf of the descion tree.
    def genLeaf(self, indices: np.ndarray, depth: int) -> LeafDict:
        target_vals = self.dataset_numpy[self.target_col][indices]
        unique, counts = np.unique(target_vals, return_counts=True)
        total = len(target_vals)
        return {
            "leaf": {k: v / total for k, v in zip(unique, counts)},
            "depth": f"{depth}#",
            "n": int(total),
        }

    def _gini(self, indices: np.ndarray) -> float:
        if len(indices) == 0:
            return 0.0
        p = np.bincount(self._y_codes[indices], minlength=len(self.class_vals)) / len(indices)
        return 1.0 - float(np.sum(p * p))

    def _split_gini(self, idx_a: np.ndarray, idx_b: np.ndarray) -> float:
        n = len(idx_a) + len(idx_b)
        return (len(idx_a) * self._gini(idx_a) + len(idx_b) * self._gini(idx_b)) / n

    def _choose_cut(self, lower_counts: np.ndarray, y_sorted: np.ndarray) -> int:
        """Index into `lower_counts` of the cut: closest to the median by count, or (with
        split_supervision >= 1) the lowest-Gini cut among those closest to the quartiles."""
        n = len(y_sorted)
        targets = [n / 2] if self.split_supervision == 0 else [n / 2, n / 4, 3 * n / 4]
        cands = []
        for t in targets:
            i = int(np.argmin(np.abs(lower_counts - t)))
            if i not in cands:
                cands.append(i)
        if len(cands) == 1:
            return cands[0]
        K = len(self.class_vals)
        best, best_gini = cands[0], np.inf
        for i in cands:
            left, right = y_sorted[: lower_counts[i]], y_sorted[lower_counts[i] :]
            g = 0.0
            for part in (left, right):
                p = np.bincount(part, minlength=K) / len(part)
                g += len(part) * (1.0 - float(np.sum(p * p)))
            if g / n < best_gini:
                best, best_gini = i, g / n
        return best

    def _split_legacy(self, feat, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Union[Real, str]]:
        """v1.5.0 split rule (median with '>' tie fallback; special case for <= 2 samples)."""
        feat_vals = self.dataset_numpy[feat][indices]

        if len(indices) > 2:
            if feat in self.numeric_cols:
                split_val = float(np.quantile(feat_vals, 0.5))  # Median
                mask = feat_vals >= split_val
                idx_a = indices[mask]
                idx_b = indices[~mask]

                if (len(idx_a) == 0) or (len(idx_b) == 0):
                    mask = feat_vals > split_val
                    idx_a = indices[mask]
                    idx_b = indices[~mask]

            else:
                # Most common value
                unique, counts = np.unique(feat_vals, return_counts=True)
                split_val = unique[np.argmax(counts)]

                mask = feat_vals == split_val
                idx_a = indices[mask]
                idx_b = indices[~mask]

            return (idx_a, idx_b, split_val)

        else:
            # Sort by feature value
            sorted_idx = np.argsort(feat_vals)
            indices = indices[sorted_idx]
            return (indices[1:], indices[:1], feat_vals[sorted_idx][1])

    def _find_split(self, feat, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Any, str]:
        """Split of the node rows on `feat`: (rows to '>=', rows to '<', split value, feat_type).

        Every split is reproduced exactly by the inference routing rule (`tree.goes_ge`).
        """
        feat_type = self.type_of_cols[feat]
        empty = np.array([], dtype=int)
        legacy = self.split_rule == "legacy" and self.split_supervision == 0
        if legacy and (feat_type == "numeric" or self.categorical_split == "mode"):
            idx_a, idx_b, split_val = self._split_legacy(feat, indices)
            return idx_a, idx_b, _py_scalar(split_val), feat_type

        vals = self.dataset_numpy[feat][indices]
        y = self._y_codes[indices]

        if feat_type == "numeric":
            order = np.argsort(vals, kind="stable")
            sorted_vals = vals[order]
            bounds = np.nonzero(sorted_vals[1:] != sorted_vals[:-1])[0]
            if bounds.size == 0:
                return indices, empty, _py_scalar(sorted_vals[0]), feat_type
            b = bounds[self._choose_cut(bounds + 1, y[order])]
            lo, hi = float(sorted_vals[b]), float(sorted_vals[b + 1])
            theta = lo + (hi - lo) / 2.0
            if not lo < theta <= hi:
                theta = hi
            mask = vals >= theta
            return indices[mask], indices[~mask], theta, feat_type

        uniq, inv, counts = np.unique(vals, return_inverse=True, return_counts=True)
        if len(uniq) < 2:
            return indices, empty, _py_scalar(uniq[0]), feat_type

        if self.categorical_split == "mode":
            mode = uniq[np.argmax(counts)]
            mask = vals == mode
            return indices[mask], indices[~mask], _py_scalar(mode), feat_type

        # Ordered categories: sort by class share, split at the count-median of that order.
        K = len(self.class_vals)
        class_counts = np.zeros((len(uniq), K))
        np.add.at(class_counts, (inv, y), 1)
        target = 1 if K == 2 else int(np.argmax(class_counts.sum(axis=0)))
        share = class_counts[:, target] / counts
        cat_order = np.lexsort((np.arange(len(uniq)), share))
        rank = np.empty(len(uniq), dtype=int)
        rank[cat_order] = np.arange(len(uniq))
        row_rank = rank[inv]
        row_order = np.argsort(row_rank, kind="stable")
        sorted_rank = row_rank[row_order]
        bounds = np.nonzero(sorted_rank[1:] != sorted_rank[:-1])[0]
        cut_rank = sorted_rank[bounds[self._choose_cut(bounds + 1, y[row_order])]]
        upper = sorted(_py_scalar(uniq[c]) for c in cat_order[cut_rank + 1 :])
        mask = row_rank > cut_rank
        return indices[mask], indices[~mask], upper, "categorical_set"

    # Splits the data during the tree's growth process.
    def splitData(self, feat, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Union[Real, str]]:
        idx_a, idx_b, split_val, _ = self._find_split(feat, indices)
        return idx_a, idx_b, split_val

    def _valid_split(self, idx_a: np.ndarray, idx_b: np.ndarray) -> bool:
        return min(len(idx_a), len(idx_b)) >= max(1, self.min_samples_split)

    def _choose_split(self, F: List[featName], idx: np.ndarray):
        """Feature and split for a node, or None; rotates the feature queue F in place."""
        if self.split_supervision == 2:
            evaluated = []
            for pos, feat in enumerate(F):
                res = self._find_split(feat, idx)
                if self._valid_split(res[0], res[1]):
                    evaluated.append((self._split_gini(res[0], res[1]), pos, feat, res))
                if pos + 1 >= self.split_lookahead and evaluated:
                    break
            if not evaluated:
                return None
            _, pos, feat, res = min(evaluated, key=lambda item: (item[0], item[1]))
            F[:] = F[pos + 1 :] + F[: pos + 1]
            return feat, res
        for _ in range(len(F)):
            feat = F[0]
            res = self._find_split(feat, idx)
            F.append(F.pop(0))
            if self._valid_split(res[0], res[1]):
                return feat, res
        return None

    def plantTree(self, indices: np.ndarray, feature_list: List[featName]) -> DecisionTreeMC:
        feature_sequence = list(feature_list)
        max_depth_reached = [1]

        def leaf(idx, depth):
            max_depth_reached[0] = max(max_depth_reached[0], depth)
            return self.genLeaf(idx, depth)

        def growTree(F: List[featName], idx: np.ndarray, depth: int = 1) -> dict:
            if (depth >= self.max_depth) or (len(np.unique(self._y_codes[idx])) == 1):
                return leaf(idx, depth)
            chosen = self._choose_split(F, idx)
            if chosen is None:
                return leaf(idx, depth)
            feat, (idx_a, idx_b, split_val, feat_type) = chosen
            return {
                feat: {
                    "split": {
                        "feat_type": feat_type,
                        "split_val": split_val,
                        "n_ge": int(len(idx_a)),
                        "n_lt": int(len(idx_b)),
                        ">=": growTree(F[:], idx_a, depth + 1),
                        "<": growTree(F[:], idx_b, depth + 1),
                    }
                }
            }

        tree = DecisionTreeMC(growTree(list(feature_list), indices), self.class_vals)
        tree.feature_sequence = feature_sequence
        tree.max_depth_reached = max_depth_reached[0]
        return tree

    # --- Validation ---

    def _tree_probs(self, Tree: DecisionTreeMC, indices: np.ndarray) -> np.ndarray:
        """(len(indices), K) class probabilities, columns ordered as `class_vals`."""
        K = len(self.class_vals)
        out = np.zeros((len(indices), K))
        pos = np.empty(self.n_samples, dtype=int)
        pos[indices] = np.arange(len(indices))
        class_pos = {c: i for i, c in enumerate(self.class_vals)}
        for leaf_node, idx in route_matrix(Tree.data, self.dataset_numpy, indices):
            probs = Tree._leaf_probs(leaf_node, self.leaf_alpha)
            vec = np.zeros(K)
            for c, p in probs.items():
                vec[class_pos[c]] = p
            out[pos[idx]] = vec
        return out

    def _score(self, Tree: DecisionTreeMC, indices: np.ndarray) -> Tuple[float, float]:
        """(accuracy, Brier score) on `indices`; accuracy is 1.0 on an empty set (v1.5.0 C++ behavior)."""
        if len(indices) == 0:
            return 1.0, 0.0
        probs = self._tree_probs(Tree, indices)
        y = self._y_codes[indices]
        acc = float(np.mean(np.argmax(probs, axis=1) == y))
        onehot = np.eye(len(self.class_vals))[y]
        brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
        return acc, brier

    # Generates the metric for validation process.
    def validationTree(self, Tree: DecisionTreeMC, indices: np.ndarray) -> float:
        return self._score(Tree, indices)[0]

    # --- Monte Carlo tree selection ---

    def _evaluate_candidate(self, Tree: DecisionTreeMC, idx_S, idx_C, slot: int, batch_id: int) -> Tuple[float, dict]:
        """Scores a candidate on the selection and scoring sets; returns (Brier score, candidate record)."""
        s_sel, brier = self._score(Tree, idx_S)
        s_score = self._score(Tree, idx_C)[0] if len(idx_C) else None
        Tree.selection_score = s_sel
        Tree.survived_score = s_score if s_score is not None else s_sel
        record = {
            "slot": int(slot),
            "batch": batch_id,
            "F": list(Tree.feature_sequence),
            "used_features": Tree.tree_feats(),
            "depth": Tree.max_depth_reached,
            "s_sel": s_sel,
            "s_score": s_score,
            "accepted": False,
        }
        return brier, record

    def _patience_action(self, best_score: float, threshold: float, tau_min: float) -> str:
        """What a slot does after `max_discard_trees` failures: 'accept' the batch best (Regime A),
        'relax' the threshold (Regime B), or 'resample' the batches when either would go below the floor."""
        if self.get_best_tree:
            return "accept" if best_score >= tau_min else "resample"
        return "resample" if threshold - self.delta_th < tau_min else "relax"

    def _run_slot(self, slot: int = 0):
        """Grows candidates for one tree slot until one is accepted.

        Returns (accepted tree, candidate records, candidate trees, candidate holdout probabilities);
        the last two are filled only for `selection="ensemble"`.
        """
        class_indices = getattr(self, "_slot_class_indices", None) or self._class_indices
        keep_candidates = self.selection == "ensemble"
        ens_idx = getattr(self, "_ens_sel_idx", None)
        tau_min = self._tau_min()
        max_attempts = self._max_attempts()

        records, cand_trees, cand_probs = [], [], []
        tree_records = {}
        # Best candidates as (key, -attempt, tree): keys compare first, then the earliest attempt wins.
        no_candidate = ((-np.inf, -np.inf), 0, None)
        global_best = no_candidate
        attempts, batch_id = 0, 0
        accepted = None

        def new_batch():
            idx_T, idx_S, idx_C = self._draw_batches(class_indices)
            return idx_T, idx_S, idx_C, self.th_start, 0, no_candidate

        idx_T, idx_S, idx_C, threshold, fails, best = new_batch()
        while True:
            attempts += 1
            F = self.sampleFeats(self.feature_cols[:])
            Tree = self.plantTree(idx_T, F)
            brier, record = self._evaluate_candidate(Tree, idx_S, idx_C, slot, batch_id)
            s_sel = Tree.selection_score
            records.append(record)
            tree_records[id(Tree)] = record
            if keep_candidates:
                cand_trees.append(Tree)
                cand_probs.append(self._tree_probs(Tree, ens_idx).astype(np.float32))

            candidate = ((s_sel, -brier if self.leaf_alpha > 0 else 0.0), -attempts, Tree)
            best = max(best, candidate, key=lambda item: item[:2])
            global_best = max(global_best, candidate, key=lambda item: item[:2])

            if s_sel >= threshold and s_sel >= tau_min:
                accepted = Tree
                break
            fails += 1
            if max_attempts is not None and attempts >= max_attempts:
                accepted = global_best[2]
                accepted.hit_attempt_cap = True
                break
            if fails >= self.max_discard_trees:
                action = self._patience_action(best[0][0], threshold, tau_min)
                if action == "accept":
                    accepted = best[2]
                    break
                if action == "resample":
                    idx_T, idx_S, idx_C, threshold, fails, best = new_batch()
                    batch_id += 1
                else:
                    threshold -= self.delta_th
                    log.info("New threshold for drop: {:.4f}".format(threshold))

        if self.got_best_tree_verbose:
            log.info("Got best tree: {:.4f}".format(accepted.selection_score))
        accepted.features = self.feature_cols
        accepted.used_features = self.tree2feats(accepted)
        accepted.attempts = attempts
        tree_records[id(accepted)]["accepted"] = True
        return accepted, records, cand_trees, cand_probs

    # _ argument is for compatible execution with thread_map
    def survivedTree(self, _=None) -> DecisionTreeMC:
        return self._run_slot(0 if _ is None else _)[0]

    def _collect(self, results) -> None:
        """Builds the forest from slot results (per-tree acceptance or ensemble selection)."""
        records = [rec for _, recs, _, _ in results for rec in recs]
        self.candidate_log.extend(records)
        if self.selection == "per_tree":
            Forest = [tree for tree, _, _, _ in results]
        else:
            cand_trees = [tree for _, _, trees, _ in results for tree in trees]
            for tree in cand_trees:
                tree.features = self.feature_cols
                tree.used_features = self.tree2feats(tree)
            probs = np.stack([p for _, _, _, ps in results for p in ps])
            y_sel = self._y_codes[self._ens_sel_idx]
            chosen = ensemble_select(
                probs, y_sel, n_select=self.n_trees, n_init=self.ensemble_init, metric=self.ensemble_metric
            )
            Forest = [cand_trees[i] for i in chosen]
            eval_idx = self._ens_score_idx if len(self._ens_score_idx) else self._ens_sel_idx
            eval_probs = np.stack([self._tree_probs(tree, eval_idx) for tree in Forest])
            self.ensemble_score = ensemble_score(eval_probs, self._y_codes[eval_idx])
        self.data.extend(Forest)
        self.survived_scores.extend([Tree.survived_score for Tree in Forest])
        self._warn_attempt_cap()

    def _prepare_slots(self) -> None:
        self._slot_class_indices = None
        self._ens_sel_idx = None
        self._ens_score_idx = None
        if self.selection == "ensemble":
            self._slot_class_indices, self._ens_sel_idx, self._ens_score_idx = self._carve_ensemble_pool()

    def fit(self, dataset: Optional[pd.DataFrame] = None, disable_progress_bar: bool = False) -> None:
        if dataset is not None:
            self.process_dataset(dataset)

        if self.dataset_numpy is None:
            raise DatasetNotFound

        # Builds the Forest (training step)
        self._prepare_slots()
        results = [
            self._run_slot(slot)
            for slot in tqdm(
                range(self.n_trees),
                disable=disable_progress_bar,
                desc="Planting the forest",
            )
        ]
        self._collect(results)

    def fitParallel(
        self,
        dataset: Optional[pd.DataFrame] = None,
        disable_progress_bar: bool = False,
        max_workers: Optional[int] = None,
    ):
        if dataset is not None:
            self.process_dataset(dataset)

        if self.dataset_numpy is None:
            raise DatasetNotFound

        self._prepare_slots()
        if not self.threaded_fit:
            results = process_map(
                self._run_slot,
                range(0, self.n_trees),
                max_workers=max_workers,
                disable=disable_progress_bar,
                desc="Planting the forest",
            )
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(
                    tqdm(
                        executor.map(self._run_slot, range(self.n_trees)),
                        total=self.n_trees,
                        disable=disable_progress_bar,
                        desc="Planting the forest",
                    )
                )
        self._collect(results)

    def sampleClass2trees(self, row: PandasSeriesRow, Class: TypeClassVal) -> List[DecisionTreeMC]:
        return [Tree for Tree in self.data if self.maxProbClas(Tree(row)) == Class]

    @property
    def trees2depths(self) -> List[List[str]]:
        return [tree.depths for tree in self.data]

    def tree2feats(self, Tree) -> List[featName]:
        return Tree.tree_feats()

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

    def featImportance(self, Forest: Optional[List[DecisionTreeMC]] = None) -> Dict[featName, float]:
        if Forest is None:
            Forest = self.data
        n_trees = len(Forest)
        importance = defaultdict(int)
        for tree in Forest:
            for feat in tree.used_features:
                importance[feat] += 1
        return {feat: count / n_trees for feat, count in importance.items()}

    def sampleClassFeatImportance(self, row: PandasSeriesRow, Class: TypeClassVal) -> Dict[featName, float]:
        return self.featImportance(self.sampleClass2trees(row=row, Class=Class))

    def featScoreMean(self, Forest: Optional[List[DecisionTreeMC]] = None) -> Dict[featName, float]:
        if Forest is None:
            Forest = self.data
        # Hadouken!!
        scores = defaultdict(list)
        for Tree in Forest:
            for feat in Tree.used_features:
                scores[feat].append(Tree.survived_score)
        return {feat: np.mean(s) for feat, s in scores.items()}

    def sampleClassFeatScoreMean(self, row: PandasSeriesRow, Class: TypeClassVal) -> Dict[featName, float]:
        return self.featScoreMean(self.sampleClass2trees(row=row, Class=Class))

    def featPairImportance(
        self, disable_progress_bar=False, Forest: Optional[List[DecisionTreeMC]] = None
    ) -> Dict[Tuple[featName, featName], float]:
        if Forest is None:
            Forest = self.data
        pair_count = defaultdict(int)
        n_trees = len(Forest)
        for Tree in tqdm(Forest, disable=disable_progress_bar, desc="Counting pair occurences"):
            used_features_in_tree = set(Tree.used_features)
            for pair in combinations(self.feature_cols, 2):
                if pair[0] in used_features_in_tree and pair[1] in used_features_in_tree:
                    pair_count[pair] += 1 / n_trees
        return dict(pair_count)

    def sampleClassFeatPairImportance(
        self, row: PandasSeriesRow, Class: TypeClassVal
    ) -> Dict[Tuple[featName, featName], float]:
        return self.featPairImportance(self.sampleClass2trees(row=row, Class=Class))

    def featCorrDataFrame(self, Forest: Optional[List[DecisionTreeMC]] = None) -> pd.DataFrame:
        N = len(self.feature_cols)
        matrix = np.zeros((N, N), dtype=np.float16)

        for feat, count in self.featImportance(Forest=Forest).items():
            idx = self.feature_cols.index(feat)
            matrix[idx][idx] = count

        for pair, count in self.featPairImportance(Forest=Forest).items():
            idxa = self.feature_cols.index(pair[0])
            idxb = self.feature_cols.index(pair[1])
            matrix[idxa][idxb], matrix[idxb][idxa] = count, count

        return pd.DataFrame(matrix, index=self.feature_cols, columns=self.feature_cols)

    def sampleClassFeatCorrDataFrame(self, row: PandasSeriesRow, Class: TypeClassVal) -> pd.DataFrame:
        return self.featCorrDataFrame(self.sampleClass2trees(row=row, Class=Class))

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
            log.warning(f"The Forest model have not the following feature(s): [{_tmp}].")
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
                # Coverage trick!
                _ = None
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
            missing_cols = []
            cols = list(dict_values.keys())
            cond = df_data_miss[cols[0]] == row[cols[0]]
            for col in cols[1:]:
                if not pd.isna(row[col]):
                    cond = cond & (df_data_miss[col] == row[col])
                    continue
                missing_cols.append(col)

            df_tmp = df_predict.loc[cond]
            df_tmp = pd.concat([pd.DataFrame(row).transpose(), df_tmp]).drop_duplicates().reset_index(drop=True)
            df_tmp["row_id"] = i
            out.append(df_tmp)

        return pd.concat(out).reset_index(drop=True)


# EOF
