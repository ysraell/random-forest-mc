"""
Forest of trees-based ensemble methods.

Random forests: extremely randomized trees with dynamic tree selection Monte Carlo based.

The module structure is the following:

"""

import logging as log
import re
from collections import defaultdict
from collections import UserDict
from collections import UserList
from hashlib import md5
from itertools import combinations
from itertools import count as itertools_count
from math import fsum
from numbers import Number
from numbers import Real
from random import randint
from random import sample
from random import shuffle
from sys import getrecursionlimit
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from .__init__ import __version__

# For backward compatibility with 3.7
# from typing import TypeAlias

typer_error_msg = "Both objects must be instances of '{}' class."

# For extract the feature names from the tree-dict.
re_feat_name = re.compile("\\'[\\w\\s]+'\\:")

# a row of pd.DataFrame.iterrows()
# dsRow: TypeAlias = pd.core.series.Series
dsRow = pd.core.series.Series

# A tree composed by a assimetric tree of dictionaries:
# TypeTree: TypeAlias = Dict
TypeTree = Dict

# Value type of classes
# TypeClassVal: TypeAlias = Any
TypeClassVal = Any  # !Review if is not forced to be str!

# Type of the leaf
# TypeLeaf: TypeAlias = Dict[TypeClassVal, float]
TypeLeaf = Dict[TypeClassVal, float]

# How to format a dict with values to fill the missing ones
featName = str
featValue = Union[str, Number]
dictValues = Dict[featName, featValue]

# Row (dsRow) or Matrix (Pandas DataFrame)
rowOrMatrix = Union[dsRow, pd.DataFrame]


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
class dictValuesAllFeaturesMissing(Exception):
    """Exception raised when all features in 'dict_values' are not found in the trained model.

    Attributes:
        message -- explanation of the error
    """

    def __init__(
        self,
        message="All features in the given dictionary 'dict_values' are not found int he trained model (forest).",
    ):
        super().__init__(message)


class DecisionTreeMC(UserDict):
    """Tree for decision. Can be used alone.
    Is originally designed to be trained and used
    with the class 'RandomForestMC'.

    Args:
        UserDict (_type_): _description_

    Raises:
        TypeError: _description_

    Returns:
        _type_: _description_
    """

    typer_error_msg = typer_error_msg.format("DecisionTreeMC")

    def __init__(
        self,
        data: dict,
        class_vals: List[TypeClassVal],
        survived_score: Optional[Real] = None,
        features: Optional[List[featName]] = None,
        used_features: Optional[List[featName]] = None,
    ) -> None:
        """_summary_

        Args:
            data (dict): _description_
            class_vals (List[TypeClassVal]): _description_
            survived_score (Optional[Real], optional): _description_. Defaults to None.
            features (Optional[List[featName]], optional): _description_. Defaults to None.
            used_features (Optional[List[featName]], optional): _description_. Defaults to None.
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

    def __call__(self, row: dsRow) -> TypeLeaf:
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
    def depths(self) -> List[str]:
        str_tree_splitted = str(self).split(" ")
        depths = []
        while str_tree_splitted:
            term = str_tree_splitted.pop(0)
            if term == "'depth':":
                depths.append(
                    int(str_tree_splitted.pop(0).split("#")[0].replace("'", ""))
                )
        return depths

    @staticmethod
    def _useTree(Tree, row: dsRow) -> TypeLeaf:
        def functionalUseTree(subTree):
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
                tree_node_split["feat_type"] == "numeric"
                and val > tree_node_split["split_val"]
            ):
                return functionalUseTree(tree_node_split[">="])
            return functionalUseTree(tree_node_split["<"])

        return functionalUseTree(Tree)

    def useTree(self, row: dsRow) -> TypeLeaf:
        Tree = self.data.copy()
        out = self._useTree(Tree, row)
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


class RandomForestMC(UserList):
    """_summary_

    Args:
        UserList (_type_): _description_

    Raises:
        TypeError: _description_
        TypeError: _description_
        TypeError: _description_
        ValueError: _description_
        ValueError: _description_
        DatasetNotFound: _description_
        DatasetNotFound: _description_

    Returns:
        _type_: _description_
    """

    typer_error_msg = typer_error_msg.format("RandomForestMC")

    def __init__(
        self,
        n_trees: int = 16,
        target_col: str = "target",
        batch_train_pclass: int = 10,
        batch_val_pclass: int = 10,
        max_discard_trees: int = 10,
        delta_th: float = 0.1,
        th_start: float = 0.9,
        get_best_tree: bool = True,
        min_feature: Optional[int] = None,
        max_feature: Optional[int] = None,
        th_decease_verbose: bool = False,
        temporal_features: bool = False,
        split_with_replace: bool = False,
        max_depth: Optional[int] = None,
        min_samples_split: int = 1,
    ) -> None:
        """_summary_

        Args:
            n_trees (int, optional): _description_. Defaults to 16.
            target_col (str, optional): _description_. Defaults to "target".
            batch_train_pclass (int, optional): _description_. Defaults to 10.
            batch_val_pclass (int, optional): _description_. Defaults to 10.
            max_discard_trees (int, optional): _description_. Defaults to 10.
            delta_th (float, optional): _description_. Defaults to 0.1.
            th_start (float, optional): _description_. Defaults to 0.9.
            get_best_tree (bool, optional): _description_. Defaults to True.
            min_feature (Optional[int], optional): _description_. Defaults to None.
            max_feature (Optional[int], optional): _description_. Defaults to None.
            th_decease_verbose (bool, optional): _description_. Defaults to False.
            temporal_features (bool, optional): _description_. Defaults to False.
            split_with_replace (bool, optional): _description_. Defaults to False.
            max_depth (Optional[int], optional): _description_. Defaults to None.
            min_samples_split (int, optional): _description_. Defaults to 1.
        """
        self.__version__ = __version__
        self.version = __version__
        self.model_version = __version__
        if th_decease_verbose:
            log.basicConfig(level=log.INFO)
        self.target_col = target_col
        self.get_best_tree = get_best_tree
        self.batch_train_pclass = batch_train_pclass
        self.batch_val_pclass = batch_val_pclass
        self._N = batch_train_pclass + batch_val_pclass
        self.min_feature = min_feature
        self.max_feature = max_feature
        self.th_start = th_start
        self.delta_th = delta_th
        self.max_discard_trees = max_discard_trees
        self.temporal_features = temporal_features
        self.n_trees = n_trees
        self.split_train_val_replace = split_with_replace
        self.dataset = None
        self.feat_types = ["numeric", "categorical"]
        self.numeric_cols = None
        self.feature_cols = None
        self.type_of_cols = None
        self.dataset = None
        self.class_vals = None
        self.reset_forest()
        self.attr_to_save = [
            "batch_train_pclass",
            "batch_val_pclass",
            "_N",
            "min_feature",
            "max_feature",
            "th_start",
            "delta_th",
            "max_discard_trees",
            "n_trees",
            "class_vals",
            "survived_scores",
            "version",
            "numeric_cols",
            "feature_cols",
            "type_of_cols",
            "target_col",
            "class_vals",
            "min_samples_split",
        ]
        self.soft_voting = False
        self.weighted_tree = False
        self.max_depth = getrecursionlimit() if max_depth is None else int(max_depth)
        self.min_samples_split = int(min_samples_split)

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
        if not isinstance(other, RandomForestMC):
            raise TypeError(self.typer_error_msg)
        return all(
            [getattr(self, att) == getattr(other, att) for att in self.attr_to_save]
        )

    def predict_proba(
        self, row_or_matrix: rowOrMatrix, prob_output: bool = True
    ) -> Union[TypeLeaf, List[TypeClassVal], List[TypeLeaf]]:
        return self.predict(row_or_matrix, prob_output)

    def predict(
        self, row_or_matrix: rowOrMatrix, prob_output: bool = False
    ) -> Union[TypeLeaf, List[TypeClassVal], List[TypeLeaf]]:
        if isinstance(row_or_matrix, dsRow):
            return self.useForest(row_or_matrix)
        if isinstance(row_or_matrix, pd.DataFrame):
            if prob_output:
                return self.testForestProbs(row_or_matrix)
            return self.testForest(row_or_matrix)
        raise TypeError(f"The input argument must be '{dsRow}' or '{pd.DataFrame}'.")

    def mergeForest(self, otherForest, N: int = -1, by: str = "add"):
        if not isinstance(otherForest, RandomForestMC):
            raise TypeError(self.typer_error_msg)
        same_model = all(
            [right in otherForest.feature_cols for right in self.feature_cols]
        ) and all([left in self.feature_cols for left in otherForest.feature_cols])
        if not same_model:
            raise ValueError("Both forests must have the same set of features.")

        same_model = all(
            [right in otherForest.class_vals for right in self.class_vals]
        ) and all([left in self.class_vals for left in otherForest.class_vals])
        if not same_model:
            raise ValueError("Both forests must have the same set of classes.")

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

    def setSoftVoting(self, set: bool = True) -> None:
        self.soft_voting = set

    def setWeightedTrees(self, set: bool = True) -> None:
        self.weighted_tree = set

    def reset_forest(self) -> None:
        self.data = []
        self.survived_scores = []

    def model2dict(self) -> dict:
        out = {attr: getattr(self, attr) for attr in self.attr_to_save}
        out["Forest"] = [Tree.tree2dict() for Tree in self.data]
        return out

    def dict2model(self, dict_model: dict) -> None:
        for attr in self.attr_to_save:
            setattr(self, attr, dict_model[attr])
        self.model_version = self.version
        self.version = self.__version__
        self.data = [
            DecisionTreeMC(
                Tree["data"],
                Tree["class_vals"],
                Tree["survived_score"],
                Tree["features"],
                Tree["used_features"],
            )
            for Tree in dict_model["Forest"]
        ]

    def validFeaturesTemporal(self):
        return all([x.split("_")[-1].isnumeric() for x in self.feature_cols])

    def drop_duplicated_trees(self) -> None:
        conds = (
            pd.DataFrame(
                [md5(str(Tree).encode("utf-8")).hexdigest() for Tree in self.data]
            )  # noqa: S303
            .duplicated()
            .to_list()
        )
        self.data = [Tree for Tree, cond in zip(self.data, conds) if cond]
        self.survived_scores = [
            score for score, cond in zip(self.survived_scores, conds) if cond
        ]

    @property
    def Forest_size(self) -> int:
        return len(self)

    @property
    def Forest(self) -> List[DecisionTreeMC]:
        return self.data

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
        self.dataset = dataset.dropna()
        self.class_vals = dataset[self.target_col].unique().tolist()

        if self.temporal_features and (not self.validFeaturesTemporal()):
            self.temporal_features = False
            log.warning(
                "Temporal features ordering disable: you do not have all orderable features!"
            )

        if not self.split_train_val_replace:
            min_class = self.dataset[self.target_col].value_counts().min()
            self._N = min(self._N, min_class)

    # Splits the data to build the decision tree.
    def split_train_val(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        idx_train = []
        idx_val = []
        for val in self.class_vals:
            idx_list = (
                self.dataset.query(f'{self.target_col} == "{val}"')
                .sample(n=self._N, replace=self.split_train_val_replace)
                .index.to_list()
            )
            idx_train.extend(idx_list[: self.batch_train_pclass])
            idx_val.extend(idx_list[self.batch_train_pclass :])

        ds_T, ds_V = (
            self.dataset.loc[idx_train].reset_index(drop=True),
            self.dataset.loc[idx_val].reset_index(drop=True),
        )
        for col in self.feature_cols:
            if ds_T[col].nunique() == 1:
                # Coverage trick!
                _ = None
                ds_T.drop(columns=col, inplace=True)
        return ds_T, ds_V

    # Sample the features.
    def sampleFeats(self, feature_cols: List[str]) -> List[featName]:
        feature_cols.remove(self.target_col)
        n_samples = randint(self.min_feature, self.max_feature)  # noqa: S311
        out = sample(feature_cols, min(len(feature_cols), n_samples))
        if not self.temporal_features:
            return out
        return sorted(out, key=lambda x: int(x.split("_")[-1]))

    # Set the leaf of the descion tree.
    def genLeaf(self, ds: pd.DataFrame, depth: int) -> TypeLeaf:
        return {
            "leaf": ds[self.target_col].value_counts(normalize=True).to_dict(),
            "depth": f"{depth}#",
        }

    # Splits the data during the tree's growth process.
    def splitData(
        self, feat, ds: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Union[Real, str]]:
        if ds.shape[0] > 2:

            if feat in self.numeric_cols:
                split_val = float(ds[feat].quantile())
                ds_a = ds.query(f"{feat} >= {split_val}").reset_index(drop=True)
                ds_b = ds.query(f"{feat} < {split_val}").reset_index(drop=True)
                if (ds_a.shape[0] == 0) or (ds_b.shape[0] == 0):
                    # I know that is a trick!
                    ds_a = ds.query(f"{feat} > {split_val}").reset_index(drop=True)
                    ds_b = ds.query(f"{feat} <= {split_val}").reset_index(drop=True)

            else:
                # Default values for Series.value_counts(
                #   normalize=False, sort=True, ascending=False,
                #   bins=None, dropna=True
                # )
                # split_val = 'most common cat value by class'
                split_val = ds[[feat, self.target_col]].value_counts().index[0][0]
                # '>=' : equal to split_val
                # '<' : not equal to split_val
                ds_a = ds.query(f'{feat} == "{split_val}"').reset_index(drop=True)
                ds_b = ds.query(f'{feat} != "{split_val}"').reset_index(drop=True)

            return (ds_a, ds_b, split_val)

        else:
            ds = ds.sort_values(feat).reset_index(drop=True)
            return (ds.loc[1].to_frame().T, ds.loc[0].to_frame().T, ds[feat].loc[1])

    def plantTree(
        self, ds_train: pd.DataFrame, feature_list: List[featName]
    ) -> DecisionTreeMC:

        # Functional process.
        def growTree(
            F: List[featName], ds: pd.DataFrame, depth: int = 1
        ) -> Union[TypeTree, TypeLeaf]:

            if (depth >= self.max_depth) or (ds[self.target_col].nunique() == 1):
                return self.genLeaf(ds, depth)

            Pass = False
            first_feat = F[0]
            while not Pass:
                feat = F[0]
                ds_a, ds_b, split_val = self.splitData(feat, ds)
                F.append(F.pop(0))
                if (ds_a.shape[0] >= self.min_samples_split) and (
                    ds_b.shape[0] >= self.min_samples_split
                ):
                    Pass = True
                elif first_feat == F[0]:
                    return self.genLeaf(ds, depth)

            return {
                feat: {
                    "split": {
                        "feat_type": self.type_of_cols[feat],
                        "split_val": split_val,
                        ">=": growTree(F[:], ds_a, depth + 1),
                        "<": growTree(F[:], ds_b, depth + 1),
                    }
                }
            }

        return DecisionTreeMC(growTree(feature_list, ds_train), self.class_vals)

    @staticmethod
    def maxProbClas(leaf: TypeLeaf) -> TypeClassVal:
        return sorted(leaf.items(), key=lambda x: x[1], reverse=True)[0][0]

    # Generates the metric for validation process.
    def validationTree(self, Tree: TypeTree, ds: pd.DataFrame) -> float:
        y_pred = [self.maxProbClas(Tree(row)) for _, row in ds.iterrows()]
        y_val = ds[self.target_col].to_list()
        return fsum([v == p for v, p in zip(y_val, y_pred)]) / len(y_pred)

    # _ argument is for compatible execution with thread_map
    def survivedTree(self, _=None) -> DecisionTreeMC:
        ds_T, ds_V = self.split_train_val()
        Threshold_for_drop = self.th_start
        dropped_trees_counter = itertools_count(1)
        # Only survived trees
        max_th_val = 0.0
        max_Tree = None
        while True:
            F = self.sampleFeats(ds_T.columns.tolist())
            Tree = self.plantTree(ds_T, F)
            th_val = self.validationTree(Tree, ds_V)
            if th_val < Threshold_for_drop:
                if self.get_best_tree:
                    if max_th_val < th_val:
                        max_Tree = Tree
                        max_th_val = th_val
            else:
                # Coverage trick!
                max_th_val = th_val
                break
            if next(dropped_trees_counter) >= self.max_discard_trees:
                if self.get_best_tree:
                    Tree = max_Tree
                    break
                else:
                    Threshold_for_drop -= self.delta_th
                    log.info(
                        "New threshold for drop: {:.4f}".format(Threshold_for_drop)
                    )

        log.info("Got best tree: {:.4f}".format(max_th_val))
        Tree.survived_score = max_th_val
        Tree.features = self.feature_cols
        Tree.used_features = self.tree2feats(Tree)
        return Tree

    def fit(
        self, dataset: Optional[pd.DataFrame] = None, disable_progress_bar: bool = False
    ) -> None:

        if dataset is not None:
            self.process_dataset(dataset)

        if self.dataset is None:
            raise DatasetNotFound

        # Builds the Forest (training step)
        Forest = []
        survived_scores = []
        for _ in tqdm(
            range(self.n_trees),
            disable=disable_progress_bar,
            desc="Planting the forest",
        ):
            Tree = self.survivedTree()
            Forest.append(Tree)
            survived_scores.append(Tree.survived_score)

        self.data.extend(Forest)
        self.survived_scores.extend(survived_scores)

    def fitParallel(
        self,
        dataset: Optional[pd.DataFrame] = None,
        disable_progress_bar: bool = False,
        max_workers: Optional[int] = None,
    ):

        if dataset is not None:
            self.process_dataset(dataset)

        if self.dataset is None:
            raise DatasetNotFound

        Tree_list = process_map(
            self.survivedTree,
            range(0, self.n_trees),
            max_workers=max_workers,
            disable=disable_progress_bar,
            desc="Planting the forest",
        )

        self.data.extend(Tree_list)
        self.survived_scores.extend([Tree.survived_score for Tree in Tree_list])

    def useForest(self, row: dsRow) -> TypeLeaf:
        if self.soft_voting:
            if self.weighted_tree:
                class_probs = defaultdict(float)
                pred_probs = [
                    (Tree(row), score)
                    for Tree, score in zip(self.data, self.survived_scores)
                ]
                for predp, score in pred_probs:
                    for class_val, prob in predp.items():
                        class_probs[class_val] += prob * score
                return {
                    class_val: class_probs[class_val] / fsum(self.survived_scores)
                    for class_val in self.class_vals
                }
            else:
                class_probs = defaultdict(float)
                pred_probs = [Tree(row) for Tree in self.data]
                for predp in pred_probs:
                    for class_val, prob in predp.items():
                        class_probs[class_val] += prob
                return {
                    class_val: class_probs[class_val] / len(pred_probs)
                    for class_val in self.class_vals
                }
        else:
            if self.weighted_tree:
                y_pred_score = [
                    (self.maxProbClas(Tree(row)), score)
                    for Tree, score in zip(self.data, self.survived_scores)
                ]
                class_scores = defaultdict(float)
                for class_val, score in y_pred_score:
                    class_scores[class_val] += score
                return {
                    class_val: class_scores[class_val] / fsum(self.survived_scores)
                    for class_val in self.class_vals
                }
            else:
                y_pred = [self.maxProbClas(Tree(row)) for Tree in self.data]
                return {
                    class_val: y_pred.count(class_val) / len(y_pred)
                    for class_val in self.class_vals
                }

    def testForest(self, ds: pd.DataFrame) -> List[TypeClassVal]:
        return [self.maxProbClas(self.useForest(row)) for _, row in ds.iterrows()]

    def testForestProbs(self, ds: pd.DataFrame) -> List[TypeLeaf]:
        return [self.useForest(row) for _, row in ds.iterrows()]

    def sampleClass2trees(self, row: dsRow, Class: TypeClassVal) -> List[TypeTree]:
        return [Tree for Tree in self.data if self.maxProbClas(Tree(row)) == Class]

    @property
    def trees2depths(self) -> List[List[str]]:
        return [tree.depths for tree in self.data]

    def tree2feats(self, Tree) -> List[featName]:
        set_keys = set(re_feat_name.findall(str(Tree)))
        set_feat_keys = set([f"'{f}':" for f in self.feature_cols])
        found_feat_keys = set_keys.intersection(set_feat_keys)
        return [feat.replace("'", "").replace(":", "") for feat in found_feat_keys]

    def featCount(
        self, Forest: Optional[List[TypeTree]] = None
    ) -> Tuple[Tuple[float, float, int, int], List[int]]:
        if Forest is None:
            Forest = self.data
        out = [len(Tree.used_features) for Tree in Forest]
        return (np.mean(out), np.std(out), min(out), max(out)), out

    def sampleClassFeatCount(
        self, row: dsRow, Class: TypeClassVal
    ) -> Tuple[Tuple[float, float, int, int], List[int]]:
        return self.featCount(self.sampleClass2trees(row=row, Class=Class))

    def featImportance(
        self, Forest: Optional[List[TypeTree]] = None
    ) -> Dict[featName, float]:
        if Forest is None:
            Forest = self.data
        n_trees = len(Forest)
        return {
            feat: fsum([f"'{feat}'" in str(Tree) for Tree in Forest]) / n_trees
            for feat in self.feature_cols
        }

    def sampleClassFeatImportance(
        self, row: dsRow, Class: TypeClassVal
    ) -> Dict[featName, float]:
        return self.featImportance(self.sampleClass2trees(row=row, Class=Class))

    def featScoreMean(
        self, Forest: Optional[List[TypeTree]] = None
    ) -> Dict[featName, float]:
        if Forest is None:
            Forest = self.data
        # Hadouken!!
        return {
            feat: np.mean(
                [
                    x
                    for x in [
                        (f"'{feat}'" in str(Tree)) * score
                        for Tree, score in zip(Forest, self.survived_scores)
                    ]
                    if x > 0
                ]
            )
            for feat in self.feature_cols
        }

    def sampleClassFeatScoreMean(
        self, row: dsRow, Class: TypeClassVal
    ) -> Dict[featName, float]:
        return self.featScoreMean(self.sampleClass2trees(row=row, Class=Class))

    def featPairImportance(
        self, disable_progress_bar=False, Forest: Optional[List[TypeTree]] = None
    ) -> Dict[Tuple[featName, featName], float]:
        if Forest is None:
            Forest = self.data
        pair_count = defaultdict(int)
        n_trees = len(Forest)
        for Tree in tqdm(
            Forest, disable=disable_progress_bar, desc="Counting pair occurences"
        ):
            for pair in combinations(self.feature_cols, 2):
                pair_count[pair] += (
                    f"'{pair[0]}'" in str(Tree) and f"'{pair[1]}'" in str(Tree)
                ) / n_trees
        return dict(pair_count)

    def sampleClassFeatPairImportance(
        self, row: dsRow, Class: TypeClassVal
    ) -> Dict[Tuple[featName, featName], float]:
        return self.featPairImportance(self.sampleClass2trees(row=row, Class=Class))

    def featCorrDataFrame(
        self, Forest: Optional[List[TypeTree]] = None
    ) -> pd.DataFrame:
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

    def sampleClassFeatCorrDataFrame(
        self, row: dsRow, Class: TypeClassVal
    ) -> pd.DataFrame:
        return self.featCorrDataFrame(self.sampleClass2trees(row=row, Class=Class))

    @staticmethod
    def _fill_row_missing(row: dsRow, dict_values: dictValues) -> pd.DataFrame:
        list_out = []
        for col, vals in dict_values.items():
            if pd.isna(row[col]):
                for val in vals:
                    _row = row.copy()
                    _row[col] = val
                    list_out.append(_row)
        if len(list_out) == 0:
            log.warning("Filling rows process: found row without missing data!")
            return None
        return pd.concat(list_out, axis=1).transpose().reset_index(drop=True)

    def _validationMissingValues(self, dict_values: dictValues) -> None:
        used_features = set()
        for Tree in self:
            used_features |= set(Tree.used_features)
        not_have_feats = set(dict_values.keys()) - used_features
        if not_have_feats:
            _tmp = ", ".join(not_have_feats)
            log.warning(
                f"The Forest model have not the following feature(s): [{_tmp}]."
            )
        if len(set(dict_values.keys())) == len(not_have_feats):
            # Coverage trick!
            _ = None
            raise dictValuesAllFeaturesMissing

    def _genFilledDataMissing(
        self, row_or_matrix: rowOrMatrix, dict_values: dictValues
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if isinstance(row_or_matrix, dsRow):
            df_data_miss = self._fill_row_missing(row_or_matrix, dict_values)
            if df_data_miss is None:
                # Coverage trick!
                _ = None
                raise MissingValuesNotFound
            row_or_matrix = (
                pd.DataFrame(row_or_matrix).transpose().reset_index(drop=True)
            )

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

    def predictMissingValues(self, row_or_matrix: rowOrMatrix, dict_values: dictValues):

        self._validationMissingValues(dict_values)

        row_or_matrix, df_data_miss = self._genFilledDataMissing(
            row_or_matrix, dict_values
        )

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
            df_tmp = (
                pd.concat([pd.DataFrame(row).transpose(), df_tmp])
                .drop_duplicates()
                .reset_index(drop=True)
            )
            df_tmp["row_id"] = i
            out.append(df_tmp)

        return pd.concat(out).reset_index(drop=True)


# EOF
