"""
Forest of trees-based ensemble methods.

Random forests: extremely randomized trees with dynamic tree selection Monte Carlo based.

The module structure is the following:

"""
from collections import defaultdict
from random import randint
from random import sample
from typing import Any
from typing import Dict
from typing import List
from typing import NewType
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm

# a row of pd.DataFrame.iterrows()
dsRow = NewType("dsRow", pd.core.series.Series)

# A tree composed by a assimetric tree of dictionaries:
TypeTree = NewType("TypeTree", Dict)

# Value type of classes
TypeClassVal = NewType("TypeClassVal", Any)

# Type of the leaf
TypeLeaf = Dict[TypeClassVal, float]


class DatasetNotFound(Exception):
    pass


class RandomForestMC:
    """ """

    def __init__(
        self,
        n_trees: int = 16,
        target_col: str = "target",
        batch_train_pclass: int = 10,
        batch_val_pclass: int = 10,
        max_discard_trees: int = 10,
        delta_th: float = 0.1,
        th_start: float = 0.9,
        min_feature: int = None,
        max_feature: int = None,
    ) -> None:
        self.target_col = target_col
        self.batch_train_pclass = batch_train_pclass
        self.batch_val_pclass = batch_val_pclass
        self._N = batch_train_pclass + batch_val_pclass
        self.min_feature = min_feature
        self.max_feature = max_feature
        self.th_start = th_start
        self.delta_th = delta_th
        self.max_discard_trees = max_discard_trees
        self.n_trees = n_trees
        self.dataset = None
        self.feat_types = ["numeric", "categorical"]

    def process_dataset(self, dataset: pd.DataFrame) -> None:
        dataset[self.target_col] = dataset[self.target_col].astype(str)
        feature_cols = [col for col in dataset.columns if col != self.target_col]
        numeric_cols = dataset.select_dtypes([np.number]).columns.to_list()
        categorical_cols = list(set(feature_cols) - set(numeric_cols))
        type_of_cols = {col: "numeric" for col in numeric_cols}
        type_of_cols.update({col: "categorical" for col in categorical_cols})

        if self.min_feature is None:
            self.min_feature = len(feature_cols) // 2

        if self.max_feature is None:
            self.max_feature = len(feature_cols)

        self.numeric_cols = numeric_cols
        self.feature_cols = feature_cols
        self.type_of_cols = type_of_cols
        self.dataset = dataset.dropna()
        self.class_vals = dataset[self.target_col].unique().tolist()

    # Splits the data to build the decision tree.
    def split_train_val(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        idx_train = []
        idx_val = []
        for val in self.class_vals:
            idx_list = (
                self.dataset.query(f'{self.target_col} == "{val}"')
                .sample(n=self._N)
                .index.to_list()
            )
            idx_train.extend(idx_list[: self.batch_train_pclass])
            idx_val.extend(idx_list[self.batch_train_pclass :])

        return (
            self.dataset.loc[idx_train].reset_index(drop=True),
            self.dataset.loc[idx_val].reset_index(drop=True),
        )

    # Sample the features.
    def sampleFeats(self) -> List[str]:
        return sample(self.feature_cols, randint(self.min_feature, self.max_feature))

    # Set the leaf of the descion tree.
    def genLeaf(self, ds) -> TypeLeaf:
        return {"leaf": ds[self.target_col].value_counts(normalize=True).to_dict()}

    # Splits the data during the tree's growth process.
    def splitData(
        self, feat, ds: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Union[int, float]]:
        if ds.shape[0] > 2:

            if feat in self.numeric_cols:
                split_val = int(ds[feat].quantile())
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

    def plantTree(self, ds_train: pd.DataFrame, feature_list: List[str]) -> TypeTree:

        # Functional process.
        def growTree(F: List[str], ds: pd.DataFrame) -> Union[TypeTree, TypeLeaf]:

            if ds[self.target_col].nunique() == 1:
                return self.genLeaf(ds)

            Pass = False
            first_feat = F[0]
            while not Pass:
                feat = F[0]
                ds_a, ds_b, split_val = self.splitData(feat, ds)
                F.append(F.pop(0))
                if (ds_a.shape[0] > 0) and (ds_b.shape[0] > 0):
                    Pass = True
                else:
                    if first_feat == F[0]:
                        return self.genLeaf(ds)

            return {
                feat: {
                    "split": {
                        "feat_type": self.type_of_cols[feat],
                        "split_val": split_val,
                        ">=": growTree(F[:], ds_a),
                        "<": growTree(F[:], ds_b),
                    }
                }
            }

        return growTree(feature_list, ds_train)

    @staticmethod
    def useTree(Tree: TypeTree, row: dsRow) -> TypeLeaf:
        while True:
            node = list(Tree.keys())[0]
            if node == "leaf":
                return Tree["leaf"]
            val = row[node]
            tree_node_split = Tree[node]["split"]
            if tree_node_split["feat_type"] == "numeric":
                Tree = (
                    tree_node_split[">="]
                    if val >= tree_node_split["split_val"]
                    else tree_node_split["<"]
                )
            Tree = (
                tree_node_split[">="]
                if val == tree_node_split["split_val"]
                else tree_node_split["<"]
            )

    @staticmethod
    def maxProbClas(leaf: TypeLeaf) -> TypeClassVal:
        return sorted(leaf.items(), key=lambda x: x[1], reverse=True)[0][0]

    # Generates the metric for validation process.
    def validationTree(self, Tree: TypeTree, ds: pd.DataFrame) -> float:
        y_pred = [self.maxProbClas(self.useTree(Tree, row)) for _, row in ds.iterrows()]
        y_val = ds[self.target_col].to_list()
        return sum([v == p for v, p in zip(y_val, y_pred)]) / len(y_pred)

    def fit(
        self, dataset: pd.DataFrame = None, disable_progress_bar: bool = False
    ) -> None:

        if dataset is not None:
            self.process_dataset(dataset)

        if self.dataset is None:
            raise DatasetNotFound(
                "Dataset not found! Please, give a dataset for functions fit() or process_dataset()."
            )

        # Builds the Forest (training step)
        Forest = []
        for t in tqdm(
            range(self.n_trees),
            disable=disable_progress_bar,
            desc="Planting the forest",
        ):
            # Builds the decision tree

            # Parallelize this loops!!
            ds_T, ds_V = self.split_train_val()
            Threshold_for_drop = self.th_start
            droped_trees = 0
            Pass = False
            # Only survived trees
            while not Pass:
                F = self.sampleFeats()
                Tree = self.plantTree(ds_T, F)
                if self.validationTree(Tree, ds_V) < Threshold_for_drop:
                    droped_trees += 1
                else:
                    Pass = True
                if droped_trees >= self.max_discard_trees:
                    Threshold_for_drop -= self.delta_th
                    droped_trees = 0

            Forest.append(Tree)

        self.Forest = Forest

    def useForest(self, row: dsRow, soft_voting: bool = False) -> TypeLeaf:
        if soft_voting:
            class_probs = defaultdict(float)
            pred_probs = [self.useTree(Tree, row) for Tree in self.Forest]
            for predp in pred_probs:
                for class_val, prob in predp.items():
                    class_probs[class_val] += prob
            return {
                class_val: class_probs[class_val] / len(pred_probs)
                for class_val in self.class_vals
            }
        else:
            y_pred = [self.maxProbClas(self.useTree(Tree, row)) for Tree in self.Forest]
            return {
                class_val: y_pred.count(class_val) / len(y_pred)
                for class_val in self.class_vals
            }

    def testForest(
        self, ds: pd.DataFrame, soft_voting: bool = False
    ) -> List[TypeClassVal]:
        return [
            self.maxProbClas(self.useForest(row, soft_voting))
            for _, row in ds.iterrows()
        ]

    def testForestProbs(
        self, ds: pd.DataFrame, soft_voting: bool = False
    ) -> List[TypeLeaf]:
        return [self.useForest(row, soft_voting) for _, row in ds.iterrows()]


# EOF
