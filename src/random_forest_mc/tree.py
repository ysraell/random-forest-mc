"""
Forest of trees-based ensemble methods.

Random forests: extremely randomized trees with dynamic tree selection Monte Carlo based.

"""

from collections import UserDict
from hashlib import md5
from math import fsum
from numbers import Real
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
import json

import numpy as np
import pandas as pd

from .__init__ import __version__
from .utils import json_encoder

typer_error_msg = "Both objects must be instances of '{}' class."

# a row of pd.DataFrame.iterrows()
PandasSeriesRow = pd.Series

# Value type of classes
TypeClassVal = Any  # !Review if is not forced to be str!

# Type of the leaf
LeafDict = Dict[TypeClassVal, float]

# How to format a dict with values to fill the missing ones
featName = str
featValue = Union[str, Real]
dictValues = Dict[featName, featValue]

# Row (dsRow) or Matrix (Pandas DataFrame)
rowOrMatrix = Union[PandasSeriesRow, pd.DataFrame, Dict[str, Any]]

# Serialization format of the tree structure.
# 1: v1.5.0 and older (no training counts in the nodes).
# 2: v1.6.0+ (split nodes store 'n_ge'/'n_lt', leaves store 'n').
TREE_FORMAT = 2

# Strategies for features missing at inference time.
MISSING_STRATEGIES = ("weighted", "uniform_leaves")

_MISSING = object()


def _row_value(row: Union[PandasSeriesRow, Dict[str, Any]], feat: featName) -> Any:
    """Returns the value of `feat` in `row`, or `_MISSING` when absent or NaN."""
    if isinstance(row, pd.Series):
        if feat not in row.index:
            return _MISSING
    elif feat not in row:
        return _MISSING
    val = row[feat]
    try:
        if pd.isna(val):
            return _MISSING
    except (TypeError, ValueError):
        pass
    return val


def goes_ge(split: dict, val: Any) -> bool:
    """Routing rule shared by training and inference: True sends `val` to the '>=' branch."""
    feat_type = split["feat_type"]
    if feat_type == "numeric":
        return val >= split["split_val"]
    if feat_type == "categorical_set":
        return val in split["split_val"]
    return val == split["split_val"]


def node_feature(node: dict) -> featName:
    """Name of the feature of a split node (the single key that is not a leaf key)."""
    return next(iter(node.keys()))


def walk_tree(node: dict):
    """Yields (feature, split dict) for every split node of a tree-dict."""
    if "leaf" in node:
        return
    feat = node_feature(node)
    split = node[feat]["split"]
    yield feat, split
    yield from walk_tree(split[">="])
    yield from walk_tree(split["<"])


def walk_leaves(node: dict):
    """Yields every leaf node of a tree-dict."""
    if "leaf" in node:
        yield node
        return
    split = node[node_feature(node)]["split"]
    yield from walk_leaves(split[">="])
    yield from walk_leaves(split["<"])


class DecisionTreeMC(UserDict):
    """A single decision tree used for making predictions.

    This class represents a single decision tree that can be used for classification tasks.
    It is designed to be part of a RandomForestMC ensemble but can also be used as a standalone model.

    Args:
        data (dict): The tree structure, represented as a nested dictionary.
        class_vals (List[TypeClassVal]): A list of all possible class values.
        survived_score (Optional[Real]): The score of the tree, used for ranking within the forest.
        features (Optional[List[featName]]): A list of all features used in the dataset.
        used_features (Optional[List[featName]]): A list of features actually used in this tree.
    """

    __slots__ = [
        "data",
        "class_vals",
        "survived_score",
        "features",
        "used_features",
        "module_version",
        "tree_format",
        "feature_sequence",
        "selection_score",
        "attempts",
        "hit_attempt_cap",
        "max_depth_reached",
        "attr_to_save",
    ]
    typer_error_msg = "Both objects must be instances of 'DecisionTreeMC' class."

    def __init__(
        self,
        data: dict,
        class_vals: List[TypeClassVal],
        survived_score: Optional[Real] = None,
        features: Optional[List[featName]] = None,
        used_features: Optional[List[featName]] = None,
        tree_format: int = TREE_FORMAT,
        feature_sequence: Optional[List[featName]] = None,
        selection_score: Optional[Real] = None,
        attempts: Optional[int] = None,
        hit_attempt_cap: bool = False,
        max_depth_reached: Optional[int] = None,
    ) -> None:
        """Initializes the DecisionTreeMC object.

        Args:
            data (dict): The dictionary representing the tree structure.
            class_vals (List[TypeClassVal]): The list of possible class values.
            survived_score (Optional[Real], optional): The survival score of the tree. Defaults to None.
            features (Optional[List[featName]], optional): The list of all features. Defaults to None.
            used_features (Optional[List[featName]], optional): The list of features used in this tree. Defaults to None.
            tree_format (int, optional): Serialization format of `data`. Defaults to the current format.
            feature_sequence (Optional[List[featName]], optional): The sampled feature queue F. Defaults to None.
            selection_score (Optional[Real], optional): Score on the selection set. Defaults to None.
            attempts (Optional[int], optional): Monte Carlo candidates grown for this slot. Defaults to None.
            hit_attempt_cap (bool, optional): Whether the slot stopped at the attempt cap. Defaults to False.
            max_depth_reached (Optional[int], optional): Depth of the deepest leaf. Defaults to None.
        """
        self.data = data
        self.class_vals = class_vals
        self.survived_score = survived_score
        self.features = features
        self.used_features = used_features
        self.module_version = __version__
        self.tree_format = tree_format
        self.feature_sequence = feature_sequence
        self.selection_score = selection_score
        self.attempts = attempts
        self.hit_attempt_cap = hit_attempt_cap
        self.max_depth_reached = max_depth_reached
        self.attr_to_save = [
            "data",
            "class_vals",
            "survived_score",
            "features",
            "used_features",
            "module_version",
            "tree_format",
            "feature_sequence",
            "selection_score",
            "attempts",
            "hit_attempt_cap",
            "max_depth_reached",
        ]

    @classmethod
    def from_dict(cls, tree_dict: dict) -> "DecisionTreeMC":
        """Builds a tree from `tree2dict()` output, accepting dicts from older versions."""
        tree = cls(
            tree_dict["data"],
            tree_dict["class_vals"],
            tree_dict.get("survived_score"),
            tree_dict.get("features"),
            tree_dict.get("used_features"),
            tree_format=tree_dict.get("tree_format", 1),
            feature_sequence=tree_dict.get("feature_sequence"),
            selection_score=tree_dict.get("selection_score"),
            attempts=tree_dict.get("attempts"),
            hit_attempt_cap=bool(tree_dict.get("hit_attempt_cap", False)),
            max_depth_reached=tree_dict.get("max_depth_reached"),
        )
        if "module_version" in tree_dict:
            tree.module_version = tree_dict["module_version"]
        return tree

    def _check_format(self, other):
        if not isinstance(other, DecisionTreeMC):
            raise TypeError(self.typer_error_msg)

    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self) -> str:
        txt = "DecisionTreeMC(survived_score={},module_version={})"
        return txt.format(self.survived_score, self.module_version)

    def __call__(self, row: PandasSeriesRow) -> LeafDict:
        return self.useTree(row)

    def __eq__(self, other) -> bool:
        self._check_format(other)
        return self.survived_score == other.survived_score

    def __gt__(self, other) -> bool:
        self._check_format(other)
        return self.survived_score > other.survived_score

    def __ge__(self, other) -> bool:
        self._check_format(other)
        return self.survived_score >= other.survived_score

    def __lt__(self, other) -> bool:
        self._check_format(other)
        return self.survived_score < other.survived_score

    def __le__(self, other) -> bool:
        self._check_format(other)
        return self.survived_score <= other.survived_score

    def tree2dict(self) -> dict:
        return {attr: getattr(self, attr) for attr in self.attr_to_save}

    @property
    def md5hexdigest(self) -> str:
        """Return the MD5 hexdigest of the tree structure for a unique signature."""
        # Sorting keys is required to have a deterministic hash
        tree_str = json.dumps(self.data, sort_keys=True, default=json_encoder)
        return md5(tree_str.encode("utf-8")).hexdigest()

    @property
    def depths(self) -> List[int]:
        return [int(str(leaf["depth"]).split("#")[0]) for leaf in walk_leaves(self.data)]

    def tree_feats(self) -> List[featName]:
        """Features used by the split nodes, in order of first appearance (depth-first)."""
        seen = []
        for feat, _ in walk_tree(self.data):
            if feat not in seen:
                seen.append(feat)
        return seen

    def leaf_for(self, row: Union[PandasSeriesRow, Dict[str, Any]]) -> Optional[dict]:
        """The leaf node reached by a row without missing features (None if one is missing)."""
        node = self.data
        while "leaf" not in node:
            feat = node_feature(node)
            split = node[feat]["split"]
            val = _row_value(row, feat)
            if val is _MISSING:
                return None
            node = split[">="] if goes_ge(split, val) else split["<"]
        return node

    def _leaf_probs(self, leaf_node: dict, leaf_alpha: float) -> LeafDict:
        """Leaf class distribution with optional Laplace smoothing (needs the leaf count 'n')."""
        leaf = leaf_node["leaf"]
        n = leaf_node.get("n")
        if leaf_alpha <= 0 or not n:
            return leaf
        K = len(self.class_vals)
        denom = n + K * leaf_alpha
        return {c: (leaf.get(c, 0.0) * n + leaf_alpha) / denom for c in self.class_vals}

    @staticmethod
    def _branch_weights(split: dict):
        n_ge = split.get("n_ge")
        n_lt = split.get("n_lt")
        if n_ge is None or n_lt is None or (n_ge + n_lt) <= 0:
            return 0.5, 0.5
        total = n_ge + n_lt
        return n_ge / total, n_lt / total

    def _weighted_dist(self, node: dict, row, weight: float, out: Dict[TypeClassVal, float], leaf_alpha: float):
        """C4.5-style marginalization: a missing split sends the row down both branches,
        weighted by the training fraction of each branch."""
        if "leaf" in node:
            for c, prob in self._leaf_probs(node, leaf_alpha).items():
                out[c] = out.get(c, 0.0) + weight * prob
            return
        feat = node_feature(node)
        split = node[feat]["split"]
        val = _row_value(row, feat)
        if val is _MISSING:
            w_ge, w_lt = self._branch_weights(split)
            self._weighted_dist(split[">="], row, weight * w_ge, out, leaf_alpha)
            self._weighted_dist(split["<"], row, weight * w_lt, out, leaf_alpha)
            return
        self._weighted_dist(split[">="] if goes_ge(split, val) else split["<"], row, weight, out, leaf_alpha)

    def _reached_leaves(self, node: dict, row, leaves: List[dict]):
        """All leaves reachable by the row (both branches at every missing split)."""
        if "leaf" in node:
            leaves.append(node)
            return
        feat = node_feature(node)
        split = node[feat]["split"]
        val = _row_value(row, feat)
        if val is _MISSING:
            self._reached_leaves(split[">="], row, leaves)
            self._reached_leaves(split["<"], row, leaves)
            return
        self._reached_leaves(split[">="] if goes_ge(split, val) else split["<"], row, leaves)

    def useTree(
        self,
        row: Union[PandasSeriesRow, Dict[str, Any]],
        missing_strategy: str = "weighted",
        leaf_alpha: float = 0.0,
    ) -> LeafDict:
        """Class distribution for a row.

        Args:
            row: The input row (pandas Series or dict). A feature is missing when it is absent or NaN.
            missing_strategy: 'weighted' (default) weights each branch of a missing split by its training
                fraction; 'uniform_leaves' averages the reached leaves uniformly (v1.5.0 C++ behavior).
            leaf_alpha: Laplace smoothing of the leaf distributions (0 disables it).
        """
        leaf_node = self.leaf_for(row)
        if leaf_node is not None:
            return self._leaf_probs(leaf_node, leaf_alpha)

        if missing_strategy == "uniform_leaves":
            leaves = []
            self._reached_leaves(self.data, row, leaves)
            out = {c: 0.0 for c in self.class_vals}
            for leaf in leaves:
                for c, prob in self._leaf_probs(leaf, leaf_alpha).items():
                    out[c] += prob / len(leaves)
        elif missing_strategy == "weighted":
            out = {c: 0.0 for c in self.class_vals}
            self._weighted_dist(self.data, row, 1.0, out, leaf_alpha)
        else:
            raise ValueError(f"Invalid missing_strategy '{missing_strategy}'. Expected one of {MISSING_STRATEGIES}.")

        total_prob = fsum(out.values())
        if total_prob > 0:
            for c in out:
                out[c] /= total_prob
        return out


def route_matrix(tree_data: dict, columns: Dict[featName, np.ndarray], indices: np.ndarray):
    """Vectorized routing of complete (no missing) rows: yields (leaf node, row indices) pairs.

    Args:
        tree_data: The tree-dict.
        columns: Mapping from feature name to a full column array.
        indices: Row indices into the column arrays.
    """
    stack = [(tree_data, indices)]
    while stack:
        node, idx = stack.pop()
        if len(idx) == 0:
            continue
        if "leaf" in node:
            yield node, idx
            continue
        feat = node_feature(node)
        split = node[feat]["split"]
        vals = columns[feat][idx]
        if split["feat_type"] == "numeric":
            mask = vals >= split["split_val"]
        elif split["feat_type"] == "categorical_set":
            mask = np.isin(vals, np.asarray(split["split_val"], dtype=vals.dtype))
        else:
            mask = vals == split["split_val"]
        stack.append((split[">="], idx[mask]))
        stack.append((split["<"], idx[~mask]))


# EOF
