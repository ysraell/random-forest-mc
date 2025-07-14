"""
Forest of trees-based ensemble methods.

Random forests: extremely randomized trees with dynamic tree selection Monte Carlo based.

"""

import logging as log
import re
from collections import defaultdict
from itertools import combinations
from itertools import count as itertools_count
from math import fsum
from numbers import Real
from random import sample as random_sample
from typing import List
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
from .tree import (
    DecisionTreeMC,
    TypeClassVal,
    LeafDict,
    rowOrMatrix,
    featName,
    PandasSeriesRow,
)


# For extract the feature names from the tree-dict.
re_feat_name = re.compile("'[\\w\\s]+'\\:")

DictValues = Dict[featName, Union[str, Real]]


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
            split_with_replace (bool, optional): Whether to split with replacement. Defaults to False.
            max_depth (Optional[int], optional): Maximum depth of the trees. Defaults to None (no limit).
            min_samples_split (int, optional): Minimum number of samples required to split an internal node. Defaults to 1.
            got_best_tree_verbose (bool, optional): Whether to log when the best tree is found. Defaults to False.
        """

        super().__init__(
            n_trees=n_trees,
            target_col=target_col,
            min_feature=min_feature,
            max_feature=max_feature,
            temporal_features=temporal_features,
        )
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
        self.max_depth = getrecursionlimit() if max_depth is None else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.got_best_tree_verbose = got_best_tree_verbose
        self.get_best_tree = get_best_tree
        self.threaded_fit = threaded_fit

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
        log.warning("Rows with missing values were dropped from the dataset.")
        self.class_vals = dataset[self.target_col].unique().tolist()

        if self.temporal_features and (not self.validFeaturesTemporal()):
            self.temporal_features = False
            log.warning("Temporal features ordering disable: you do not have all orderable features!")

        min_class = self.dataset[self.target_col].value_counts().min()
        self._N = min(self._N, min_class)

    # Splits the data to build the decision tree.
    def split_train_val(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        idx_train = []
        idx_val = []
        for val in self.class_vals:
            idx_list = (
                self.dataset.query(f'{self.target_col} == "{val}"').sample(n=self._N, replace=False).index.to_list()
            )
            idx_train.extend(idx_list[: self.batch_train_pclass])
            idx_val.extend(idx_list[self.batch_train_pclass :])  # noqa E203

        ds_T, ds_V = (
            self.dataset.loc[idx_train].reset_index(drop=True),
            self.dataset.loc[idx_val].reset_index(drop=True),
        )
        for col in self.feature_cols:
            if ds_T[col].nunique() == 1:
                ds_T.drop(columns=col, inplace=True)
        return ds_T, ds_V

    # Sample the features.
    def sampleFeats(self, feature_cols: List[str]) -> List[featName]:
        feature_cols.remove(self.target_col)
        n_samples = np.random.randint(self.min_feature, self.max_feature + 1)
        out = random_sample(feature_cols, min(len(feature_cols), n_samples))
        if not self.temporal_features:
            return out
        return sorted(out, key=lambda x: int(x.split("_")[-1]))

    # Set the leaf of the descion tree.
    def genLeaf(self, ds: pd.DataFrame, depth: int) -> LeafDict:
        return {
            "leaf": ds[self.target_col].value_counts(normalize=True).to_dict(),
            "depth": f"{depth}#",
        }

    # Splits the data during the tree's growth process.
    def splitData(self, feat, ds: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Union[Real, str]]:
        if ds.shape[0] > 2:
            if feat in self.numeric_cols:
                split_val = float(ds[feat].quantile())
                ds_a = ds.query(f"{feat} >= {split_val}").reset_index(drop=True)
                ds_b = ds.query(f"{feat} < {split_val}").reset_index(drop=True)
                if (ds_a.shape[0] == 0) or (ds_b.shape[0] == 0):
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

    def plantTree(self, ds_train: pd.DataFrame, feature_list: List[featName]) -> DecisionTreeMC:
        # Functional process.
        def growTree(F: List[featName], ds: pd.DataFrame, depth: int = 1) -> Union[DecisionTreeMC, LeafDict]:
            if (depth >= self.max_depth) or (ds[self.target_col].nunique() == 1):
                return self.genLeaf(ds, depth)

            Pass = False
            first_feat = F[0]
            while not Pass:
                feat = F[0]
                ds_a, ds_b, split_val = self.splitData(feat, ds)
                F.append(F.pop(0))
                if (ds_a.shape[0] >= self.min_samples_split) and (ds_b.shape[0] >= self.min_samples_split):
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

    # Generates the metric for validation process.
    def validationTree(self, Tree: DecisionTreeMC, ds: pd.DataFrame) -> float:
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
                max_th_val = th_val
                break
            if next(dropped_trees_counter) >= self.max_discard_trees:
                if self.get_best_tree:
                    Tree = max_Tree
                    break
                else:
                    Threshold_for_drop -= self.delta_th
                    log.info("New threshold for drop: {:.4f}".format(Threshold_for_drop))
        if self.got_best_tree_verbose:
            log.info("Got best tree: {:.4f}".format(max_th_val))
        Tree.survived_score = max_th_val
        Tree.features = self.feature_cols
        Tree.used_features = self.tree2feats(Tree)
        return Tree

    def fit(self, dataset: Optional[pd.DataFrame] = None, disable_progress_bar: bool = False) -> None:
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

        if not self.threaded_fit:
            Tree_list = process_map(
                self.survivedTree,
                range(0, self.n_trees),
                max_workers=max_workers,
                disable=disable_progress_bar,
                desc="Planting the forest",
            )
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                Tree_list = list(
                    tqdm(
                        executor.map(self.survivedTree, range(self.n_trees)),
                        total=self.n_trees,
                        disable=disable_progress_bar,
                        desc="Planting the forest",
                    )
                )

        self.data.extend(Tree_list)
        self.survived_scores.extend([Tree.survived_score for Tree in Tree_list])

    def sampleClass2trees(self, row: PandasSeriesRow, Class: TypeClassVal) -> List[DecisionTreeMC]:
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
        for Tree, score in zip(Forest, self.survived_scores):
            for feat in Tree.used_features:
                scores[feat].append(score)
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
                    _row = row.copy()
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
