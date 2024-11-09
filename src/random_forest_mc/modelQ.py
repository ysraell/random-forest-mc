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
    """Decision Tree implementation with Monte Carlo methods.

    This class implements a decision tree that can be used independently or as part
    of a Random Forest ensemble. It supports both numerical and categorical features
    and handles missing values through Monte Carlo methods.

    Attributes:
        data (dict): The tree structure stored as nested dictionaries
        class_vals (List[TypeClassVal]): List of possible class values
        survived_score (Real): Tree's survival score in the forest
        features (List[str]): List of all available features
        used_features (List[str]): Features actually used in the tree
        module_version (str): Version of the module
    """

    __slots__ = ["data", "class_vals", "survived_score", "features", "used_features", "module_version", "attr_to_save"]

    def __init__(
        self,
        data: dict,
        class_vals: List[TypeClassVal],
        survived_score: Optional[Real] = None,
        features: Optional[List[featName]] = None,
        used_features: Optional[List[featName]] = None,
    ) -> None:
        """Initialize a new decision tree.

        Args:
            data: The tree structure as nested dictionaries
            class_vals: Possible class values for classification
            survived_score: Tree's survival score in the forest
            features: Available features for splitting
            used_features: Features actually used in the tree
        """
        super().__init__()
        self.data = data
        self.class_vals = class_vals
        self.survived_score = survived_score
        self.features = features or []
        self.used_features = used_features or []
        self.module_version = __version__
        self.attr_to_save = ["data", "class_vals", "survived_score", "features", "used_features", "module_version"]

    def _validate_tree_comparison(self, other: "DecisionTreeMC") -> None:
        """Validate that the other object is a DecisionTreeMC instance.

        Args:
            other: Object to compare with

        Raises:
            TypeError: If other is not a DecisionTreeMC instance
        """
        if not isinstance(other, DecisionTreeMC):
            raise TypeError(f"Comparison only supported between DecisionTreeMC instances")

    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self) -> str:
        return f"DecisionTreeMC(survived_score={self.survived_score}, module_version={self.module_version})"

    def __call__(self, row: dsRow) -> TypeLeaf:
        """Make the tree callable for predictions.

        Args:
            row: Input features as a pandas Series

        Returns:
            Predicted class probabilities
        """
        return self.predict(row)

    def predict(self, row: dsRow) -> TypeLeaf:
        """Predict class probabilities for a single input row.

        Args:
            row: Input features as a pandas Series

        Returns:
            Dictionary mapping class labels to probabilities
        """
        return self._traverse_tree(self.data, row)

    def _traverse_tree(self, tree: Dict, row: dsRow) -> TypeLeaf:
        """Recursively traverse the tree to make predictions.

        Args:
            tree: Current node in the tree
            row: Input features

        Returns:
            Leaf node probabilities or averaged probabilities for missing values
        """

        def traverse_node(subtree: Dict) -> Union[TypeLeaf, List]:
            node = next(iter(subtree))
            if node == "leaf":
                return subtree["leaf"]

            split_info = subtree[node]["split"]
            if node not in row.index:
                return [traverse_node(split_info[">="]), traverse_node(split_info["<"])]

            value = row[node]
            if value >= split_info["split_val"] or (
                split_info["feat_type"] == "numeric" and value > split_info["split_val"]
            ):
                return traverse_node(split_info[">="])
            return traverse_node(split_info["<"])

        result = traverse_node(tree)
        if isinstance(result, dict):
            return result

        # Handle missing value cases by averaging multiple paths
        return self._average_predictions(result)

    def _average_predictions(self, predictions: List) -> TypeLeaf:
        """Average predictions from multiple paths (used for missing values).

        Args:
            predictions: List of prediction dictionaries

        Returns:
            Averaged and normalized predictions
        """

        def flatten(lst):
            for item in lst:
                if isinstance(item, list):
                    yield from flatten(item)
                else:
                    yield item

        leaves = list(flatten(predictions))
        # Calculate average probabilities
        avg_probs = {
            class_val: sum(leaf.get(class_val, 0) for leaf in leaves) / len(leaves) for class_val in self.class_vals
        }

        # Normalize probabilities
        total = sum(avg_probs.values())
        return {class_val: prob / total for class_val, prob in avg_probs.items()}

    @property
    def md5hexdigest(self) -> str:
        """Calculate MD5 hash of the tree structure."""
        return md5(str(self).encode("utf-8")).hexdigest()

    @property
    def depths(self) -> List[int]:
        """Extract all depth values from the tree structure."""
        depth_pattern = re.compile(r"'depth': '(\d+)")
        return [int(match) for match in depth_pattern.findall(str(self))]

    def to_dict(self) -> dict:
        """Convert tree instance to a dictionary for serialization."""
        return {attr: getattr(self, attr) for attr in self.attr_to_save}

    # Comparison methods
    def __eq__(self, other: "DecisionTreeMC") -> bool:
        self._validate_tree_comparison(other)
        return self.survived_score == other.survived_score

    def __lt__(self, other: "DecisionTreeMC") -> bool:
        self._validate_tree_comparison(other)
        return self.survived_score < other.survived_score

    def __le__(self, other: "DecisionTreeMC") -> bool:
        self._validate_tree_comparison(other)
        return self.survived_score <= other.survived_score


from collections import UserList
from typing import Optional, Union, List
import logging as log
from sys import getrecursionlimit
from random import shuffle
import numpy as np
import pandas as pd
from .types import TypeLeaf, TypeClassVal, rowOrMatrix, dsRow


class RandomForestMC(UserList):
    """A Random Forest implementation using Monte Carlo methods for classification tasks.

    This class implements an ensemble of decision trees with Monte Carlo sampling
    for handling missing values and feature selection.

    Attributes:
        data (List[DecisionTreeMC]): List of decision trees in the forest
        target_col (str): Name of the target column
        n_trees (int): Number of trees in the forest
        batch_train_pclass (int): Number of training samples per class
        batch_val_pclass (int): Number of validation samples per class
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
    ) -> None:
        """Initialize the Random Forest.

        Args:
            n_trees: Number of trees in the forest
            target_col: Name of the target column
            batch_train_pclass: Number of training samples per class
            batch_val_pclass: Number of validation samples per class
            max_discard_trees: Maximum number of trees to discard
            delta_th: Threshold delta for tree selection
            th_start: Initial threshold value
            get_best_tree: Whether to select the best performing tree
            min_feature: Minimum number of features to consider
            max_feature: Maximum number of features to consider
            th_decease_verbose: Enable verbose logging for threshold decrease
            temporal_features: Whether to use temporal feature ordering
            split_with_replace: Whether to use replacement in splitting
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
        """
        super().__init__()
        self._initialize_version()
        self._setup_logging(th_decease_verbose)
        self._initialize_parameters(
            target_col=target_col,
            get_best_tree=get_best_tree,
            batch_train_pclass=batch_train_pclass,
            batch_val_pclass=batch_val_pclass,
            min_feature=min_feature,
            max_feature=max_feature,
            th_start=th_start,
            delta_th=delta_th,
            max_discard_trees=max_discard_trees,
            temporal_features=temporal_features,
            n_trees=n_trees,
            split_with_replace=split_with_replace,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        self._initialize_dataset_attributes()
        self.reset_forest()

    def _initialize_version(self) -> None:
        """Initialize version-related attributes."""
        self.__version__ = __version__
        self.version = __version__
        self.model_version = __version__

    def _setup_logging(self, verbose: bool) -> None:
        """Configure logging settings."""
        if verbose:
            log.basicConfig(level=log.INFO)

    def _initialize_parameters(self, **kwargs) -> None:
        """Initialize model parameters."""
        for key, value in kwargs.items():
            setattr(self, key, value)

        self._N = kwargs["batch_train_pclass"] + kwargs["batch_val_pclass"]
        self.max_depth = getrecursionlimit() if kwargs["max_depth"] is None else int(kwargs["max_depth"])
        self.min_samples_split = int(kwargs["min_samples_split"])

        # Initialize voting parameters
        self.soft_voting = False
        self.weighted_tree = False

    def _initialize_dataset_attributes(self) -> None:
        """Initialize dataset-related attributes."""
        self.dataset = None
        self.feat_types = ["numeric", "categorical"]
        self.numeric_cols = None
        self.feature_cols = None
        self.type_of_cols = None
        self.class_vals = None

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

    def predict_proba(
        self, row_or_matrix: rowOrMatrix, prob_output: bool = True
    ) -> Union[TypeLeaf, List[TypeClassVal], List[TypeLeaf]]:
        """Predict class probabilities for input data.

        Args:
            row_or_matrix: Input features as a single row or matrix
            prob_output: Whether to return probabilities

        Returns:
            Predicted probabilities or class labels
        """
        return self.predict(row_or_matrix, prob_output)

    def predict(
        self, row_or_matrix: rowOrMatrix, prob_output: bool = False
    ) -> Union[TypeLeaf, List[TypeClassVal], List[TypeLeaf]]:
        """Make predictions for input data.

        Args:
            row_or_matrix: Input features as a single row or matrix
            prob_output: Whether to return probabilities

        Returns:
            Predictions as probabilities or class labels

        Raises:
            TypeError: If input type is invalid
        """
        if isinstance(row_or_matrix, dsRow):
            return self._predict_single(row_or_matrix)
        if isinstance(row_or_matrix, pd.DataFrame):
            return self._predict_batch(row_or_matrix, prob_output)
        raise TypeError(f"Input must be '{dsRow}' or '{pd.DataFrame}'.")

    def _predict_single(self, row: dsRow) -> TypeLeaf:
        """Make prediction for a single input row."""
        return self.useForest(row)

    def _predict_batch(self, matrix: pd.DataFrame, prob_output: bool) -> Union[List[TypeClassVal], List[TypeLeaf]]:
        """Make predictions for a batch of inputs."""
        return self.testForestProbs(matrix) if prob_output else self.testForest(matrix)

    def merge_forest(self, other_forest: "RandomForestMC", n_trees: int = -1, method: str = "add") -> None:
        """Merge another forest into this one.

        Args:
            other_forest: Forest to merge with
            n_trees: Number of trees to keep after merging
            method: Merging method ('add', 'score', or 'random')

        Raises:
            TypeError: If other_forest is not a RandomForestMC
            ValueError: If forests have different class values
        """
        self._validate_merge(other_forest)

        if method == "add":
            self.data.extend(other_forest.data)
        elif method == "score":
            self._merge_by_score(other_forest, n_trees)
        elif method == "random":
            self._merge_random(other_forest, n_trees)

        self.survived_scores = [tree.survived_score for tree in self.data]

    def _validate_merge(self, other_forest: "RandomForestMC") -> None:
        """Validate merge compatibility."""
        if not isinstance(other_forest, RandomForestMC):
            raise TypeError(self.typer_error_msg)

        if not (set(self.class_vals) == set(other_forest.class_vals)):
            raise ValueError("Both forests must have the same set of classes.")

    def _merge_by_score(self, other_forest: "RandomForestMC", n_trees: int) -> None:
        """Merge forests by selecting top scoring trees."""
        combined = self.data + other_forest.data
        self.data = sorted(combined, reverse=True)[:n_trees]

    def _merge_random(self, other_forest: "RandomForestMC", n_trees: int) -> None:
        """Merge forests by random selection."""
        combined = self.data + other_forest.data
        shuffle(combined)
        self.data = combined[:n_trees]

    def process_dataset(self, dataset: pd.DataFrame) -> None:
        """Process and validate the input dataset.

        Args:
            dataset: Input DataFrame to process
        """
        self._prepare_target(dataset)
        self._identify_feature_types(dataset)
        self._set_feature_limits()
        self._validate_temporal_features()
        self.dataset = dataset.dropna()
        self.class_vals = dataset[self.target_col].unique().tolist()

    def _prepare_target(self, dataset: pd.DataFrame) -> None:
        """Prepare target column."""
        dataset[self.target_col] = dataset[self.target_col].astype(str)

    def _identify_feature_types(self, dataset: pd.DataFrame) -> None:
        """Identify numeric and categorical features."""
        feature_cols = [col for col in dataset.columns if col != self.target_col]
        numeric_cols = dataset.select_dtypes([np.number]).columns.tolist()
        categorical_cols = list(set(feature_cols) - set(numeric_cols))

        self.type_of_cols = {
            **{col: "numeric" for col in numeric_cols},
            **{col: "categorical" for col in categorical_cols},
        }
        self.numeric_cols = numeric_cols
        self.feature_cols = feature_cols

    def _set_feature_limits(self) -> None:
        """Set minimum and maximum feature limits."""
        if self.min_feature is None:
            self.min_feature = 2
        if self.max_feature is None:
            self.max_feature = len(self.feature_cols)

    def _validate_temporal_features(self) -> None:
        """Validate temporal feature configuration."""
        if self.temporal_features and not self._are_features_temporal():
            self.temporal_features = False
            log.warning("Temporal features disabled: not all features are orderable!")

    def _are_features_temporal(self) -> bool:
        """Check if features are temporal."""
        return all(x.split("_")[-1].isnumeric() for x in self.feature_cols)


# EOF
