"""
Forest of trees-based ensemble methods.

Random forests: extremely randomized trees with dynamic tree selection Monte Carlo based.

"""

import logging as log
import re
from collections import defaultdict
from collections import UserList
from math import ceil, fsum, log2
from random import shuffle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map
from .tree import (
    DecisionTreeMC,
    PandasSeriesRow,
    TypeClassVal,
    LeafDict,
    rowOrMatrix,
    MISSING_STRATEGIES,
    TREE_FORMAT,
)

from .__init__ import __version__

# For extract the feature names from the tree-dict.
re_feat_name = re.compile("\\'[\\w\\s]+'\\:")

SPLIT_RULES = ("midpoint", "legacy")
FEATURE_SAMPLINGS = ("uniform", "effective_depth")
SELECTIONS = ("per_tree", "ensemble")
ENSEMBLE_METRICS = ("balanced_accuracy", "logloss")
CATEGORICAL_SPLITS = ("mode", "ordered")
TREE_WEIGHTINGS = ("score", "samme")

# Hyperparameters introduced in v1.6.0 and the values that reproduce v1.5.0.
LEGACY_V150 = {
    "split_rule": "legacy",
    "missing_strategy": "uniform_leaves",
    "tau_floor_eps": None,
    "max_tree_attempts": None,
}


class BaseRandomForestMC(UserList):
    """Base class for Random Forest Monte Carlo.

    This class extends UserList to manage a collection of DecisionTreeMC objects,
    providing core functionalities for forest-based predictions and operations.
    """

    typer_error_msg = "Both objects must be instances of 'RandomForestMC' class."

    def __init__(
        self,
        n_trees: int = 16,
        target_col: str = "target",
        min_feature: Optional[int] = None,
        max_feature: Optional[int] = None,
        temporal_features: bool = False,
    ) -> None:
        """Initializes the BaseRandomForestMC object.

        Args:
            n_trees (int, optional): The number of trees in the forest. Defaults to 16.
            target_col (str, optional): The name of the target column. Defaults to "target".
            min_feature (Optional[int], optional): Minimum number of features to consider for each tree. Defaults to None.
            max_feature (Optional[int], optional): Maximum number of features to consider for each tree. Defaults to None.
            temporal_features (bool, optional): Whether to consider temporal features. Defaults to False.
        """
        self.__version__ = __version__
        self.version = __version__
        self.model_version = __version__
        self.target_col = target_col
        self.min_feature = min_feature
        self.max_feature = max_feature
        self.temporal_features = temporal_features
        self.n_trees = n_trees
        self.feat_types = ["numeric", "categorical"]
        self.numeric_cols = None
        self.feature_cols = None
        self.type_of_cols = None
        self.class_vals = None
        self.data = []
        self.survived_scores = []
        self.attr_to_save = [
            "min_feature",
            "max_feature",
            "n_trees",
            "class_vals",
            "survived_scores",
            "version",
            "numeric_cols",
            "feature_cols",
            "type_of_cols",
            "target_col",
            "class_vals",
        ]
        # Saved when present; absent in dicts from v1.5.0 and older.
        self.attr_to_save_optional = [
            "tree_format",
            "split_rule",
            "missing_strategy",
            "tau_floor_eps",
            "max_tree_attempts",
            "val_pool_pclass",
            "score_split",
            "feature_sampling",
            "n_train_frac",
            "n_train_cap",
            "selection",
            "split_supervision",
            "split_lookahead",
            "categorical_split",
            "leaf_alpha",
            "class_priors",
            "class_freqs",
            "tree_weighting",
            "decision_threshold",
            "positive_class",
            "ensemble_score",
            "L_eff",
        ]
        self.soft_voting = False
        self.weighted_tree = False
        self.tree_format = TREE_FORMAT
        self.class_freqs = None
        self.decision_threshold = None
        self.positive_class = None
        self.ensemble_score = None
        self.L_eff = None
        self.candidate_log = []
        self._init_v160_params()

    def _init_v160_params(
        self,
        split_rule: str = "midpoint",
        missing_strategy: str = "weighted",
        tau_floor_eps: Optional[float] = 0.05,
        max_tree_attempts: Union[int, str, None] = "auto",
        val_pool_pclass: Optional[int] = None,
        score_split: float = 0.0,
        feature_sampling: str = "uniform",
        n_train_frac: Optional[float] = None,
        n_train_cap: Optional[int] = None,
        selection: str = "per_tree",
        ensemble_pool_pclass: int = 100,
        ensemble_init: int = 3,
        ensemble_metric: str = "balanced_accuracy",
        split_supervision: int = 0,
        split_lookahead: int = 2,
        categorical_split: str = "mode",
        leaf_alpha: float = 0.0,
        class_priors: Union[None, str, Dict[TypeClassVal, float]] = None,
        tree_weighting: str = "score",
        legacy_v150: bool = False,
    ) -> None:
        """Sets and validates the hyperparameters introduced in v1.6.0.

        `legacy_v150=True` restores every v1.5.0 behavior changed by the bug fixes
        (split rule, missing-value traversal and the threshold floor).
        """
        if legacy_v150:
            split_rule = LEGACY_V150["split_rule"]
            missing_strategy = LEGACY_V150["missing_strategy"]
            tau_floor_eps = LEGACY_V150["tau_floor_eps"]
            max_tree_attempts = LEGACY_V150["max_tree_attempts"]

        def _check(name, value, options):
            if value not in options:
                raise ValueError(f"Invalid {name} '{value}'. Expected one of {options}.")

        _check("split_rule", split_rule, SPLIT_RULES)
        _check("missing_strategy", missing_strategy, MISSING_STRATEGIES)
        _check("feature_sampling", feature_sampling, FEATURE_SAMPLINGS)
        _check("selection", selection, SELECTIONS)
        _check("ensemble_metric", ensemble_metric, ENSEMBLE_METRICS)
        _check("categorical_split", categorical_split, CATEGORICAL_SPLITS)
        _check("tree_weighting", tree_weighting, TREE_WEIGHTINGS)
        _check("split_supervision", split_supervision, (0, 1, 2))
        if not 0.0 <= score_split < 1.0:
            raise ValueError("score_split must be in [0, 1).")
        if leaf_alpha < 0:
            raise ValueError("leaf_alpha must be >= 0.")
        if tau_floor_eps is not None and tau_floor_eps < 0:
            raise ValueError("tau_floor_eps must be >= 0 or None.")
        if isinstance(class_priors, str) and class_priors != "train":
            raise ValueError("class_priors must be None, 'train' or a dict of class frequencies.")

        self.split_rule = split_rule
        self.missing_strategy = missing_strategy
        self.tau_floor_eps = tau_floor_eps
        self.max_tree_attempts = max_tree_attempts
        self.val_pool_pclass = val_pool_pclass
        self.score_split = float(score_split)
        self.feature_sampling = feature_sampling
        self.n_train_frac = n_train_frac
        self.n_train_cap = n_train_cap
        self.selection = selection
        self.ensemble_pool_pclass = int(ensemble_pool_pclass)
        self.ensemble_init = int(ensemble_init)
        self.ensemble_metric = ensemble_metric
        self.split_supervision = int(split_supervision)
        self.split_lookahead = max(1, int(split_lookahead))
        self.categorical_split = categorical_split
        self.leaf_alpha = float(leaf_alpha)
        self.class_priors = class_priors
        self.tree_weighting = tree_weighting

    # --- Training configuration shared by both engines ---

    def _max_attempts(self) -> Optional[int]:
        """Cap on Monte Carlo candidates per slot (None means unlimited, the v1.5.0 behavior)."""
        if self.max_tree_attempts == "auto":
            return None if self.tau_floor_eps is None else 10 * max(1, self.max_discard_trees)
        return None if self.max_tree_attempts is None else int(self.max_tree_attempts)

    def _tau_min(self) -> float:
        """Floor of the acceptance threshold: chance level (1/K) plus `tau_floor_eps`."""
        if self.tau_floor_eps is None:
            return -np.inf
        return 1.0 / len(self.class_vals) + self.tau_floor_eps

    def _batch_sizes(self, min_class_count: int) -> Tuple[int, int, int]:
        """Per-class sizes of the training batch, selection set and scoring set."""
        n_train = self.batch_train_pclass
        if self.n_train_frac is not None:
            # Grows with the data (never below batch_train_pclass), keeping batch_val_pclass rows for validation.
            scaled = int(self.n_train_frac * min_class_count)
            if self.n_train_cap is not None:
                scaled = min(scaled, int(self.n_train_cap))
            scaled = min(scaled, min_class_count - self.batch_val_pclass)
            n_train = max(n_train, scaled)
        n_train = max(1, min(n_train, min_class_count))
        pool_pc = self.batch_val_pclass if self.val_pool_pclass is None else int(self.val_pool_pclass)
        n_pool = max(0, min(pool_pc, min_class_count - n_train))
        n_score = int(self.score_split * n_pool)
        if n_score >= n_pool and n_pool > 0:
            n_score = n_pool - 1
        return n_train, n_pool - n_score, n_score

    def _ensemble_pool_size(self, min_class_count: int) -> int:
        if self.selection != "ensemble":
            return 0
        return max(1, min(self.ensemble_pool_pclass, min_class_count // 2))

    def _set_class_freqs(self, target_values: pd.Series) -> None:
        counts = target_values.value_counts()
        self.class_freqs = {str(c): float(counts.get(c, 0)) / float(len(target_values)) for c in self.class_vals}

    def _effective_depth(self, n_train: int) -> int:
        """L_eff = ceil(log2(K * n_train)) + 1: depth at which median halving reaches pure leaves."""
        return int(ceil(log2(max(2, len(self.class_vals) * n_train)))) + 1

    def _resolved_priors(self) -> Optional[Dict[TypeClassVal, float]]:
        if self.class_priors is None:
            return None
        if isinstance(self.class_priors, str):
            return self.class_freqs
        return self.class_priors

    # --- Container behavior ---

    def __repr__(self) -> str:
        txt = "{}(len(Forest)={},n_trees={},model_version={},module_version={})"
        return txt.format(
            self.__class__.__name__,
            len(self.data),
            self.n_trees,
            self.model_version,
            self.version,
        )

    def __eq__(self, other):
        if not isinstance(other, BaseRandomForestMC):
            raise TypeError(self.typer_error_msg)
        return all([getattr(self, att) == getattr(other, att) for att in self.attr_to_save])

    def predict_proba(self, row_or_matrix: rowOrMatrix, prob_output: bool = True) -> Union[LeafDict, List[LeafDict]]:
        return self.predict(row_or_matrix, prob_output)

    def predict(
        self, row_or_matrix: rowOrMatrix, prob_output: bool = False
    ) -> Union[LeafDict, List[TypeClassVal], List[LeafDict]]:
        if isinstance(row_or_matrix, PandasSeriesRow):
            return self.useForest(row_or_matrix)
        if isinstance(row_or_matrix, pd.DataFrame):
            if prob_output:
                return self.testForestProbs(row_or_matrix)
            return self.testForest(row_or_matrix)
        raise TypeError("The input argument must be a Pandas Series or a Pandas DataFrame.")

    def _check_tree_formats(self, trees: List[DecisionTreeMC], allow_mixed_formats: bool) -> None:
        formats = {getattr(Tree, "tree_format", 1) for Tree in trees}
        if len(formats) > 1 and not allow_mixed_formats:
            raise ValueError(
                f"Trees with different serialization formats {sorted(formats)} (thresholds and hashes differ "
                "between versions). Pass allow_mixed_formats=True to combine them anyway."
            )

    def mergeForest(self, otherForest, N: int = -1, by: str = "add", allow_mixed_formats: bool = False):
        if not isinstance(otherForest, BaseRandomForestMC):
            raise TypeError(self.typer_error_msg)

        same_classes_right = all([right in otherForest.class_vals for right in self.class_vals])
        same_classes_left = all([left in self.class_vals for left in otherForest.class_vals])
        same_features_right = all([right in otherForest.feature_cols for right in self.feature_cols])
        same_features_left = all([left in self.feature_cols for left in otherForest.feature_cols])
        same_model = all(
            [
                same_classes_right,
                same_classes_left,
                same_features_right,
                same_features_left,
            ]
        )

        if not same_model:
            raise ValueError("Both forests must have the same set of features and classes.")

        self._check_tree_formats(self.data + otherForest.data, allow_mixed_formats)

        if by == "add":
            self.data.extend(otherForest.data)

        if by == "score":
            data = self.data + otherForest.data
            self.data = sorted(data)[::-1][:N]

        if by == "random":
            data = self.data + otherForest.data
            shuffle(data)
            self.data = data[:N]

        self.survived_scores = [Tree.survived_score for Tree in self.data]

    def validFeaturesTemporal(self):
        return all([x.split("_")[-1].isnumeric() for x in self.feature_cols])

    def setSoftVoting(self, set: bool = True) -> None:
        self.soft_voting = set

    def setWeightedTrees(self, set: bool = True) -> None:
        self.weighted_tree = set

    def reset_forest(self) -> None:
        self.data = []
        self.survived_scores = []
        self.candidate_log = []

    def model2dict(self, include_candidates: bool = False) -> dict:
        out = {attr: getattr(self, attr) for attr in self.attr_to_save}
        out.update({attr: getattr(self, attr, None) for attr in self.attr_to_save_optional})
        out["Forest"] = [Tree.tree2dict() for Tree in self.data]
        if include_candidates:
            out["candidate_log"] = self.candidate_log
        return out

    def dict2model(self, dict_model: dict, add: bool = False) -> None:
        new_trees = [DecisionTreeMC.from_dict(Tree) for Tree in dict_model["Forest"]]
        for attr in self.attr_to_save:
            if attr in dict_model and attr != "survived_scores":
                setattr(self, attr, dict_model[attr])
        for attr in self.attr_to_save_optional:
            if attr in dict_model:
                setattr(self, attr, dict_model[attr])
        self.tree_format = dict_model.get("tree_format", 1)
        if "candidate_log" in dict_model:
            self.candidate_log = dict_model["candidate_log"]
        self.model_version = self.version
        self.version = self.__version__
        self.data = self.data + new_trees if add else new_trees
        self.survived_scores = [Tree.survived_score for Tree in self.data]

    def drop_duplicated_trees(self, allow_mixed_formats: bool = False) -> int:
        self._check_tree_formats(self.data, allow_mixed_formats)
        conds = pd.DataFrame([Tree.md5hexdigest for Tree in self.data]).duplicated().to_list()
        self.data = [Tree for Tree, cond in zip(self.data, conds) if not cond]
        self.survived_scores = [score for score, cond in zip(self.survived_scores, conds) if not cond]
        return len(conds) - sum(conds)

    @property
    def Forest_size(self) -> int:
        return len(self)

    @property
    def Forest(self) -> List[DecisionTreeMC]:
        return self.data

    @staticmethod
    def maxProbClas(leaf: LeafDict) -> TypeClassVal:
        return sorted(leaf.items(), key=lambda x: x[1], reverse=True)[0][0]

    # --- Voting ---

    def _tree_weights(self) -> np.ndarray:
        """Per-tree voting weights: the survival scores, or SAMME log-odds of them."""
        scores = np.asarray(self.survived_scores, dtype=float)
        if self.tree_weighting == "samme":
            s = np.clip(scores, 1e-3, 1 - 1e-3)
            weights = np.log(s / (1 - s)) + np.log(max(1, len(self.class_vals) - 1))
            weights = np.maximum(weights, 0.0)
        else:
            weights = scores
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        return weights

    def _apply_priors(self, dist: LeafDict) -> LeafDict:
        """Prior correction P(c|x) ∝ p(c|x)·π_c for forests trained on class-balanced batches."""
        priors = self._resolved_priors()
        if not priors:
            return dist
        out = {c: p * priors.get(str(c), priors.get(c, 0.0)) for c, p in dist.items()}
        total = fsum(out.values())
        if total <= 0:
            return dist
        return {c: p / total for c, p in out.items()}

    def _decide(self, dist: LeafDict) -> TypeClassVal:
        """Class decision: tuned threshold on the positive class for binary targets, otherwise argmax."""
        if self.decision_threshold is not None and self.positive_class in dist and len(dist) == 2:
            if dist[self.positive_class] >= self.decision_threshold:
                return self.positive_class
            return next(c for c in dist if c != self.positive_class)
        return self.maxProbClas(dist)

    def useForest(self, row: PandasSeriesRow) -> LeafDict:
        def tree_out(Tree):
            return Tree.useTree(row, self.missing_strategy, self.leaf_alpha)

        weights = self._tree_weights() if self.weighted_tree else np.ones(len(self.data))
        total = fsum(weights)
        class_scores = defaultdict(float)
        for Tree, w in zip(self.data, weights):
            if self.soft_voting:
                for class_val, prob in tree_out(Tree).items():
                    class_scores[class_val] += prob * w
            else:
                class_scores[self.maxProbClas(tree_out(Tree))] += w
        dist = {class_val: class_scores[class_val] / total for class_val in self.class_vals}
        return self._apply_priors(dist)

    def testForest(self, ds: pd.DataFrame) -> List[TypeClassVal]:
        return [self._decide(self.useForest(row)) for _, row in ds.iterrows()]

    def _testForest_func(self, row: PandasSeriesRow):
        return self._decide(self.useForest(row))

    def testForestParallel(
        self,
        ds: pd.DataFrame,
        max_workers: Optional[int] = None,
        chunksize: Optional[int] = None,
    ) -> List[TypeClassVal]:
        ds_iterator = [row for _, row in ds.iterrows()]
        chunksize = int(np.ceil(len(ds_iterator) / (2 * (max_workers or 1)))) if chunksize is None else chunksize
        return process_map(
            self._testForest_func,
            ds_iterator,
            desc="Testing the forest",
            max_workers=max_workers,
            chunksize=chunksize,
        )

    def testForestProbs(self, ds: pd.DataFrame) -> List[LeafDict]:
        return [self.useForest(row) for _, row in ds.iterrows()]

    def testForestProbsParallel(
        self,
        ds: pd.DataFrame,
        max_workers: Optional[int] = None,
        chunksize: Optional[int] = None,
    ) -> List[TypeClassVal]:
        ds_iterator = [row for _, row in ds.iterrows()]
        chunksize = int(np.ceil(len(ds_iterator) / (2 * (max_workers or 1)))) if chunksize is None else chunksize
        return process_map(
            self.useForest,
            ds_iterator,
            desc="Testing the forest",
            max_workers=max_workers,
            chunksize=chunksize,
        )

    # --- Decision threshold ---

    def tune_decision_threshold(
        self, dataset: pd.DataFrame, positive_class: Optional[TypeClassVal] = None
    ) -> float:
        """Tunes the binary decision threshold for balanced accuracy on held-out data.

        Args:
            dataset: Held-out rows including the target column.
            positive_class: Class thresholded on. Defaults to the minority training class.

        Returns:
            The selected threshold, which `predict` uses from now on.
        """
        if len(self.class_vals) != 2:
            raise ValueError("tune_decision_threshold supports binary targets only.")
        if positive_class is None:
            if self.class_freqs:
                positive_class = min(self.class_vals, key=lambda c: self.class_freqs.get(str(c), 0.0))
            else:
                positive_class = self.class_vals[1]
        y = dataset[self.target_col].astype(str).to_numpy()
        self.decision_threshold = None
        probs = self.predict_proba(dataset.drop(columns=[self.target_col]))
        p = np.array([dist[positive_class] for dist in probs])
        is_pos = y == str(positive_class)
        candidates = np.unique(np.concatenate([p, [0.5]]))
        best_thr, best_score = 0.5, -1.0
        for thr in candidates:
            pred_pos = p >= thr
            tpr = (pred_pos & is_pos).sum() / max(1, is_pos.sum())
            tnr = (~pred_pos & ~is_pos).sum() / max(1, (~is_pos).sum())
            score = (tpr + tnr) / 2
            if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and abs(thr - 0.5) < abs(best_thr - 0.5)):
                best_thr, best_score = float(thr), score
        self.decision_threshold = best_thr
        self.positive_class = positive_class
        return best_thr

    # --- Diagnostics and candidate-based analysis ---

    def depthReport(self) -> pd.DataFrame:
        """Per tree: sampled cardinality k, realized depth, used features, and whether the used features
        are exactly the first levels of the feature queue F (the one-feature-per-level structure)."""
        rows = []
        for Tree in self.data:
            seq = Tree.feature_sequence or []
            depth = Tree.max_depth_reached if Tree.max_depth_reached is not None else max(Tree.depths, default=1)
            used = set(Tree.used_features or [])
            levels = max(0, depth - 1)
            prefix = seq[: min(len(seq), levels)] if seq else []
            rows.append(
                {
                    "k": len(seq),
                    "max_depth_reached": depth,
                    "n_used": len(used),
                    "used_is_prefix": bool(seq) and used == set(prefix),
                    "unused_tail": max(0, len(seq) - levels),
                    "L_eff": self.L_eff,
                }
            )
        return pd.DataFrame(rows)

    def featImportanceMC(
        self, prefix_len: Optional[int] = None, n_boot: int = 200, seed: Optional[int] = 0
    ) -> pd.DataFrame:
        """Feature importance from every Monte Carlo candidate (not only the survivors).

        The feature sequences are drawn uniformly at random, so 'f is among the first `prefix_len`
        positions of F' is a randomized treatment. Returns, per feature, the difference in mean
        candidate score with and without the treatment, with a cluster (per slot) bootstrap CI.
        """
        from .importance import randomized_importance

        return randomized_importance(
            self.candidate_log,
            self.feature_cols,
            prefix_len=prefix_len or self.L_eff or 5,
            n_boot=n_boot,
            seed=seed,
        )

    def featPairImportanceMC(
        self, prefix_len: Optional[int] = None, alpha: float = 1.0, max_pair_features: int = 30
    ) -> Dict[str, Any]:
        """Ridge regression of the candidate score on feature-in-prefix indicators and their pairs."""
        from .importance import ridge_importance

        return ridge_importance(
            self.candidate_log,
            self.feature_cols,
            prefix_len=prefix_len or self.L_eff or 5,
            alpha=alpha,
            pairwise=True,
            max_pair_features=max_pair_features,
        )

    def _warn_attempt_cap(self) -> None:
        n_cap = sum(1 for Tree in self.data if getattr(Tree, "hit_attempt_cap", False))
        if n_cap:
            log.warning(
                f"{n_cap} of {len(self.data)} tree slots hit max_tree_attempts without reaching the "
                f"threshold floor ({self._tau_min():.3f}); the best candidate was kept for each."
            )


# EOF
