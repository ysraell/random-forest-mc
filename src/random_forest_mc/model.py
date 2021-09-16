"""
Forest of trees-based ensemble methods.

Random forests: extremely randomized trees with dynamic tree selection Monte Carlo based.

The module structure is the following:

"""
import logging as log
from collections import defaultdict
from hashlib import md5
from itertools import combinations
from random import randint
from random import sample
from typing import Any
from typing import Dict
from typing import List
from typing import NewType
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from tqdm.contrib.concurrent import thread_map

from .__init__ import __version__


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
        th_decease_verbose=False,
    ) -> None:
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
        self.n_trees = n_trees
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
            "Forest",
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
        txt = "RandomForestMC(len(Forest)={},n_trees={},model_version={},module_version={})"
        return txt.format(
            len(self.Forest), self.n_trees, self.model_version, self.version
        )

    def setSoftVoting(self, set: bool = True) -> None:
        self.soft_voting = set

    def setWeightedTrees(self, set: bool = True) -> None:
        self.weighted_tree = set

    def reset_forest(self) -> None:
        self.Forest = []
        self.survived_scores = []

    def model2dict(self) -> dict:
        return {attr: getattr(self, attr) for attr in self.attr_to_save}

    def dict2model(self, dict_model: dict) -> None:
        for attr in self.attr_to_save:
            setattr(self, attr, dict_model[attr])
        self.model_version = self.version
        self.version = self.__version__

    def addTrees(self, Forest_Score: List[Tuple[TypeTree, float]]) -> None:
        for Tree, survived_score in Forest_Score:
            self.survived_scores.append(survived_score)
            self.Forest.append(Tree)

    def drop_duplicated_trees(self) -> None:
        conds = (
            pd.DataFrame(
                [
                    md5(str(Tree).encode("utf-8")).hexdigest() for Tree in self.Forest
                ]  # noqa: S303
            )
            .duplicated()
            .to_list()
        )
        self.Forest = [Tree for Tree, cond in zip(self.Forest, conds) if cond]
        self.survived_scores = [
            score for score, cond in zip(self.survived_scores, conds) if cond
        ]

    @property
    def Forest_size(self) -> int:
        return len(self.Forest)

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
        return sample(
            self.feature_cols, randint(self.min_feature, self.max_feature)  # noqa: S311
        )

    # Set the leaf of the descion tree.
    def genLeaf(self, ds) -> TypeLeaf:
        return {"leaf": ds[self.target_col].value_counts(normalize=True).to_dict()}

    # Splits the data during the tree's growth process.
    def splitData(
        self, feat, ds: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Union[int, float, str]]:
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
            else:
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

    def survivedTree(
        self, _=None
    ) -> Tuple[
        TypeTree, float
    ]:  # _ argument is for compatible execution with thread_map
        ds_T, ds_V = self.split_train_val()
        Threshold_for_drop = self.th_start
        dropped_trees = 0
        # Only survived trees
        max_th_val = 0.0
        max_Tree = None
        while True:
            F = self.sampleFeats()
            Tree = self.plantTree(ds_T, F)
            th_val = self.validationTree(Tree, ds_V)
            if th_val < Threshold_for_drop:
                dropped_trees += 1
                if self.get_best_tree:
                    if max_th_val < th_val:
                        max_Tree = Tree
                        max_th_val = th_val
            else:
                # Coverage trick!
                _ = None
                break
            if dropped_trees >= self.max_discard_trees:
                if self.get_best_tree:
                    Tree = max_Tree
                    Threshold_for_drop = max_th_val
                    break
                else:
                    Threshold_for_drop -= self.delta_th
                log.info("New threshold for drop: {:.4f}".format(Threshold_for_drop))
                dropped_trees = 0

        log.info("Got best tree: {:.4f}".format(Threshold_for_drop))
        return Tree, Threshold_for_drop

    def fit(
        self, dataset: Optional[pd.DataFrame] = None, disable_progress_bar: bool = False
    ) -> None:

        if dataset is not None:
            self.process_dataset(dataset)

        if self.dataset is None:
            raise DatasetNotFound(
                "Dataset not found! Please, give a dataset for functions fit() or process_dataset()."
            )

        # Builds the Forest (training step)
        Forest = []
        survived_scores = []
        for _ in tqdm(
            range(self.n_trees),
            disable=disable_progress_bar,
            desc="Planting the forest",
        ):
            Tree, Threshold_for_drop = self.survivedTree()
            Forest.append(Tree)
            survived_scores.append(Threshold_for_drop)

        self.Forest.extend(Forest)
        self.survived_scores.extend(survived_scores)

    def fitParallel(
        self,
        dataset: Optional[pd.DataFrame] = None,
        disable_progress_bar: bool = False,
        max_workers: Optional[int] = None,
        thread_parallel_method: bool = False,
    ):

        if dataset is not None:
            self.process_dataset(dataset)

        if self.dataset is None:
            raise DatasetNotFound(
                "Dataset not found! Please, give a dataset for functions fit() or process_dataset()."
            )

        # Builds the Forest (training step)
        if thread_parallel_method:
            func_map = thread_map
        else:
            func_map = process_map

        out = func_map(
            self.survivedTree,
            range(0, self.n_trees),
            max_workers=max_workers,
            disable=disable_progress_bar,
            desc="Planting the forest",
        )

        Tree_Threshold_for_drop_list = list(zip(*out))
        self.Forest.extend(Tree_Threshold_for_drop_list[0])
        self.survived_scores.extend(Tree_Threshold_for_drop_list[1])

    def useForest(self, row: dsRow) -> TypeLeaf:
        if self.soft_voting:
            if self.weighted_tree:
                class_probs = defaultdict(float)
                pred_probs = [
                    (self.useTree(Tree, row), score)
                    for Tree, score in zip(self.Forest, self.survived_scores)
                ]
                for (predp, score) in pred_probs:
                    for class_val, prob in predp.items():
                        class_probs[class_val] += prob * score
                return {
                    class_val: class_probs[class_val] / sum(self.survived_scores)
                    for class_val in self.class_vals
                }
            else:
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
            if self.weighted_tree:
                y_pred_score = [
                    (self.maxProbClas(self.useTree(Tree, row)), score)
                    for Tree, score in zip(self.Forest, self.survived_scores)
                ]
                class_scores = defaultdict(float)
                for class_val, score in y_pred_score:
                    class_scores[class_val] += score
                return {
                    class_val: class_scores[class_val] / sum(self.survived_scores)
                    for class_val in self.class_vals
                }
            else:
                y_pred = [
                    self.maxProbClas(self.useTree(Tree, row)) for Tree in self.Forest
                ]
                return {
                    class_val: y_pred.count(class_val) / len(y_pred)
                    for class_val in self.class_vals
                }

    def testForest(self, ds: pd.DataFrame) -> List[TypeClassVal]:
        return [self.maxProbClas(self.useForest(row)) for _, row in ds.iterrows()]

    def testForestProbs(self, ds: pd.DataFrame) -> List[TypeLeaf]:
        return [self.useForest(row) for _, row in ds.iterrows()]

    def tree2feats(self, Tree) -> List[str]:
        return [feat for feat in self.feature_cols if f"'{feat}'" in str(Tree)]

    def sampleClass2trees(self, row: dsRow, Class: TypeClassVal) -> List[TypeTree]:
        return [
            Tree
            for Tree in self.Forest
            if self.maxProbClas(self.useTree(Tree, row)) == Class
        ]

    def featCount(
        self, Forest: Optional[List[TypeTree]] = None
    ) -> Tuple[Tuple[float, float, int, int], List[int]]:
        if Forest is None:
            Forest = self.Forest
        out = [len(self.tree2feats(Tree)) for Tree in Forest]
        return (np.mean(out), np.std(out), min(out), max(out)), out

    def sampleClassFeatCount(
        self, row: dsRow, Class: TypeClassVal
    ) -> Tuple[Tuple[float, float, int, int], List[int]]:
        return self.featCount(self.sampleClass2trees(row=row, Class=Class))

    def featImportance(
        self, Forest: Optional[List[TypeTree]] = None
    ) -> Dict[str, float]:
        if Forest is None:
            Forest = self.Forest
        n_trees = len(Forest)
        return {
            feat: sum([f"'{feat}'" in str(Tree) for Tree in Forest]) / n_trees
            for feat in self.feature_cols
        }

    def sampleClassFeatImportance(
        self, row: dsRow, Class: TypeClassVal
    ) -> Dict[str, float]:
        return self.featImportance(self.sampleClass2trees(row=row, Class=Class))

    def featScoreMean(
        self, Forest: Optional[List[TypeTree]] = None
    ) -> Dict[str, float]:
        if Forest is None:
            Forest = self.Forest
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
    ) -> Dict[str, float]:
        return self.featScoreMean(self.sampleClass2trees(row=row, Class=Class))

    def featPairImportance(
        self, disable_progress_bar=False, Forest: Optional[List[TypeTree]] = None
    ) -> Dict[Tuple[str, str], float]:
        if Forest is None:
            Forest = self.Forest
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
    ) -> Dict[Tuple[str, str], float]:
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


# EOF
