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
rowOrMatrix = Union[PandasSeriesRow, pd.DataFrame]


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
    ) -> None:
        """Initializes the DecisionTreeMC object.

        Args:
            data (dict): The dictionary representing the tree structure.
            class_vals (List[TypeClassVal]): The list of possible class values.
            survived_score (Optional[Real], optional): The survival score of the tree. Defaults to None.
            features (Optional[List[featName]], optional): The list of all features. Defaults to None.
            used_features (Optional[List[featName]], optional): The list of features used in this tree. Defaults to None.
        """
        self.data = data
        self.class_vals = class_vals
        self.survived_score = survived_score
        self.features = features
        self.used_features = used_features
        self.module_version = __version__
        self.attr_to_save = [
            "data",
            "class_vals",
            "survived_score",
            "features",
            "used_features",
            "module_version",
        ]

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
    def depths(self) -> List[str]:
        str_tree_splitted = str(self).split(" ")
        depths = []
        while str_tree_splitted:
            term = str_tree_splitted.pop(0)
            if term == "'depth':":
                depths.append(int(str_tree_splitted.pop(0).split("#")[0].replace("'", "")))
        return depths

    @staticmethod
    def _useTree(tree, row: PandasSeriesRow) -> LeafDict:
        def functionalUseTree(subTree) -> LeafDict:
            node = list(subTree.keys())[0]
            if node == "leaf":
                return subTree["leaf"]
            tree_node_split = subTree[node]["split"]
            if node not in row.index:
                return [
                    functionalUseTree(tree_node_split[">="]),
                    functionalUseTree(tree_node_split["<"]),
                ]
            val = row[node]
            if val == tree_node_split["split_val"] or (
                tree_node_split["feat_type"] == "numeric" and val > tree_node_split["split_val"]
            ):
                return functionalUseTree(tree_node_split[">="])
            return functionalUseTree(tree_node_split["<"])

        return functionalUseTree(tree)

    def useTree(self, row: PandasSeriesRow) -> LeafDict:
        tree = self.data.copy()
        out = self._useTree(tree, row)
        if isinstance(out, dict):
            return out
        leafes = []

        def popLeafs(LeafList):
            if isinstance(LeafList, dict):
                leafes.append(LeafList)
                return
            for Leaf in LeafList:
                return popLeafs(Leaf)

        popLeafs(out)

        outLeaf = {c: 0 for c in self.class_vals}
        for leaf in leafes:
            for c, prob in leaf.items():
                outLeaf[c] += prob
        for c in self.class_vals:
            outLeaf[c] /= len(leafes)

        total_prob = fsum(outLeaf.values())
        for c in self.class_vals:
            outLeaf[c] /= total_prob

        return outLeaf


# EOF
