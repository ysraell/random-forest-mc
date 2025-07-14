"""
Forest of trees-based ensemble methods.

Random forests: extremely randomized trees with dynamic tree selection Monte Carlo based.

"""

import re
from collections import defaultdict
from collections import UserList
from math import fsum
from random import shuffle
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map
from .tree import DecisionTreeMC, PandasSeriesRow, TypeClassVal, LeafDict, rowOrMatrix

from .__init__ import __version__

# For extract the feature names from the tree-dict.
re_feat_name = re.compile("\\'[\\w\\s]+'\\:")


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
        self.soft_voting = False
        self.weighted_tree = False

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

    def mergeForest(self, otherForest, N: int = -1, by: str = "add"):
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

    def model2dict(self) -> dict:
        out = {attr: getattr(self, attr) for attr in self.attr_to_save}
        out["Forest"] = [Tree.tree2dict() for Tree in self.data]
        return out

    def dict2model(self, dict_model: dict, add: bool = False) -> None:
        new_trees = [
            DecisionTreeMC(
                Tree["data"],
                Tree["class_vals"],
                Tree["survived_score"],
                Tree["features"],
                Tree["used_features"],
            )
            for Tree in dict_model["Forest"]
        ]
        if add:
            self.data.extend(new_trees)
            self.survived_scores.extend(dict_model["survived_scores"])
        for attr in self.attr_to_save:
            setattr(self, attr, dict_model[attr])
        self.model_version = self.version
        self.version = self.__version__
        self.data = new_trees

    def drop_duplicated_trees(self) -> int:
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

    def useForest(self, row: PandasSeriesRow) -> LeafDict:
        if self.soft_voting:
            if self.weighted_tree:
                class_probs = defaultdict(float)
                pred_probs = [(Tree(row), score) for Tree, score in zip(self.data, self.survived_scores)]
                for predp, score in pred_probs:
                    for class_val, prob in predp.items():
                        class_probs[class_val] += prob * score
                return {class_val: class_probs[class_val] / fsum(self.survived_scores) for class_val in self.class_vals}
            else:
                class_probs = defaultdict(float)
                pred_probs = [Tree(row) for Tree in self.data]
                for predp in pred_probs:
                    for class_val, prob in predp.items():
                        class_probs[class_val] += prob
                return {class_val: class_probs[class_val] / len(pred_probs) for class_val in self.class_vals}
        else:
            if self.weighted_tree:
                y_pred_score = [
                    (self.maxProbClas(Tree(row)), score) for Tree, score in zip(self.data, self.survived_scores)
                ]
                class_scores = defaultdict(float)
                for class_val, score in y_pred_score:
                    class_scores[class_val] += score
                return {
                    class_val: class_scores[class_val] / fsum(self.survived_scores) for class_val in self.class_vals
                }
            else:
                y_pred = [self.maxProbClas(Tree(row)) for Tree in self.data]
                return {class_val: y_pred.count(class_val) / len(y_pred) for class_val in self.class_vals}

    def testForest(self, ds: pd.DataFrame) -> List[TypeClassVal]:
        return [self.maxProbClas(self.useForest(row)) for _, row in ds.iterrows()]

    def _testForest_func(self, row: PandasSeriesRow):
        return self.maxProbClas(self.useForest(row))

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


# EOF
