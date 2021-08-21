from typing import Dict, TypeVar, Any, List
import pandas as pd
import numpy as np
from random import randint, sample

# a row of pd.DataFrame.iterrows() 
dsRow = TypeVar('dsRow', pd.core.series.Series)

# A tree composed by a assimetric tree of dictionaries:
TypeTree = TypeVar('TypeTree', Dict[Any]) 

class DatasetNotFound(Exception):
    pass

class RandomForestMC:
    """
    [DOC COPIED FROM SKLEARN!! Just to use as model.]
    A random forest classifier.
    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting.
    The sub-sample size is controlled with the `max_samples` parameter if
    `bootstrap=True` (default), otherwise the whole dataset is used to build
    each tree.
    Read more in the :ref:`User Guide <forest>`.
    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
        .. versionchanged:: 0.22
           The default value of ``n_estimators`` changed from 10 to 100
           in 0.22.
    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for fractions.
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for fractions.
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.
        .. versionadded:: 0.19
    min_impurity_split : float, default=None
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.
        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` has changed from 1e-7 to 0 in 0.23 and it
           will be removed in 1.0 (renaming of 0.25).
           Use ``min_impurity_decrease`` instead.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
        Only available if bootstrap=True.
    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.
    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.
    verbose : int, default=0
        Controls the verbosity when fitting and predicting.
    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.
    class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts, \
            default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.
        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.
        .. versionadded:: 0.22
    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.
        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0, 1)`.
        .. versionadded:: 0.22
    Attributes
    ----------
    base_estimator_ : DecisionTreeClassifier
        The child estimator template used to create the collection of fitted
        sub-estimators.
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.
    classes_ : ndarray of shape (n_classes,) or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).
    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.
    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.
    oob_decision_function_ : ndarray of shape (n_samples, n_classes)
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN. This attribute exists
        only when ``oob_score`` is True.
    See Also
    --------
    DecisionTreeClassifier, ExtraTreesClassifier
    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.
    References
    ----------
    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = RandomForestClassifier(max_depth=2, random_state=0)
    >>> clf.fit(X, y)
    RandomForestClassifier(...)
    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]
    """
    def __init__(self,
        n_trees: int = 16,
        target_col: str = 'Class',
        batch_train_pclass: int = 10,
        batch_val_pclass: int = 10,
        max_discard_trees: int = 10,
        delta_th: float = 0.1,
        th_start: float = 0.9,
        min_feature: int = None,
        max_feature: int = None
    ) -> None:
        self.target_col = target_col
        self.batch_train_pclass = batch_train_pclass
        self.batch_val_pclass = batch_val_pclass
        self._N = batch_train_pclass+batch_val_pclass
        self.min_feature = min_feature
        self.max_feature = max_feature
        self.th_start = th_start
        self.delta_th = delta_th
        self.max_discard_trees = max_discard_trees
        self.n_trees = n_trees
        self.dataset = None

    def process_dataset(self, dataset: pd.DataFrame):
        feature_cols = [col for col in dataset.columns if col != self.target_col]
        numeric_cols = dataset.select_dtypes([np.number]).columns.to_list()
        categorical_cols = list(set(feature_cols)-set(numeric_cols))
        type_of_cols = {col: 'numeric' for col in numeric_cols}
        type_of_cols.update({col: 'categorical' for col in categorical_cols})

        if self.min_feature is None:
            self.min_feature = len(feature_cols) // 2

        if self.max_feature is None:
            self.max_feature = len(feature_cols)

        self.feature_cols = feature_cols
        self.type_of_cols = type_of_cols
        self.dataset = dataset
        self.class_vals = dataset[self.target_col].unique().tolist() 

    # Splits the data to build the decision tree.
    def split_train_val(self):
        idx_train = []
        idx_val = []
        for val in self.class_vals:
            idx_list = self.dataset.query(f'{self.target_col} == {val}').sample(n=self._N).index.to_list()
            idx_train.extend(idx_list[:self._N])
            idx_val.extend(idx_list[self._N:])
        return idx_train, idx_val

    # Sample the features.
    def sampleFeats(self):
        return sample(self.feature_cols, randint(self.min_feature, self.max_feature))

    # Set the leaf of the descion tree.
    def genLeaf(self, ds):
        return {
            'leaf' : ds[self.target_col].mode()[0]
        }

    # Splits the data during the tree's growth process.
    def splitData(feat, ds: pd.DataFrame):
        if ds.shape[0] > 2:
            split_val = int(ds[feat].quantile())
            ds_a = ds.query(f'{feat} >= {split_val}').reset_index(drop=True)
            ds_b = ds.query(f'{feat} < {split_val}').reset_index(drop=True)
            if (ds_a.shape[0] > 0) and (ds_b.shape[0] > 0):
                return (
                    ds_a,
                    ds_b,
                    split_val
                )
            # I know that is a trick!
            ds_a = ds.query(f'{feat} > {split_val}').reset_index(drop=True)
            ds_b = ds.query(f'{feat} <= {split_val}').reset_index(drop=True)
            return (
                ds_a,
                ds_b,
                split_val
            )
                
        ds = ds.sort_values(feat).reset_index(drop=True)
        return (
                ds.loc[1].to_frame().T,
                ds.loc[0].to_frame().T,
                ds[feat].loc[1]
            )
    def plantTree(self, ds_train: pd.DataFrame, feature_list: List[str]):

        # Functional process.          
        def growTree(F: List[str], ds: pd.DataFrame):

            if ds[self.target_col].nunique() == 1:
                return self.genLeaf(ds)

            Pass = False
            first_feat = F[0]
            while not Pass:
                feat = F[0]
                ds_a, ds_b, split_val = self.splitData(feat,ds)
                F.append(F.pop(0))
                if (ds_a.shape[0] > 0) and (ds_b.shape[0] > 0):
                    Pass = True
                else:
                    if first_feat == F[0]:
                        return self.genLeaf(ds)
            
            return {
                    feat: {
                        'split': {
                            'split_val': split_val,
                            '>=' : growTree(F[:],ds_a),
                            '<' : growTree(F[:],ds_b)
                        }
                    }
                }

        return growTree(feature_list, ds_train)

    @staticmethod
    def useTree(Tree: TypeTree, row: dsRow):
        while True:
            node = list(Tree.keys())[0]
            if node == 'leaf':
                return Tree['leaf']
            val = row[node]
            Tree = Tree[node]['split']['>='] if val >= Tree[node]['split']['split_val'] else Tree[node]['split']['<']

    # Generates the metric for validation process.
    def validationTree(self, Tree: TypeTree, ds: pd.DataFrame):
        y_pred = [self.useTree(Tree, row) for _,row in ds.iterrows()]
        y_val = ds[self.target_col].to_list()
        return sum([v == p for v,p in zip(y_val,y_pred)])/len(y_pred)

    def fit(self, dataset: pd.DataFrame = None):

        if dataset is not None:
            self.process_dataset(dataset)
        
        if self.dataset is None:
            raise DatasetNotFound('Dataset not found! Please, give a dataset for functions fit() or process_dataset().')

        # Builds the Forest (training step)
        Forest = []
        for t in range(self.n_trees):
            # Builds the decision tree
            idx_train, idx_val = self.split_train_val()
            ds_T = self.dataset.loc[idx_train].reset_index(drop=True)
            ds_V = self.dataset.loc[idx_val].reset_index(drop=True)
            Threshold_for_drop = self.th_start
            droped_trees = 0
            Pass = False
            # Only survived trees
            while not Pass:
                F = self.sampleFeats()
                Tree = self.plantTree(ds_T, F)
                if self.validationTree(Tree,ds_V) < Threshold_for_drop:
                    droped_trees += 1
                else:
                    Pass = True
                if droped_trees >= self.max_discard_trees:
                    Threshold_for_drop -= self.delta_th
                    droped_trees = 0
            Forest.append(Tree)

        self.Forest = Forest

    def useForest(self, row: dsRow):
        y_pred = [self.useTree(Tree, row) for Tree in self.Forest]
        return (y_pred.count(0)/len(y_pred), y_pred.count(1)/len(y_pred))

    def testForest(self, ds: pd.DataFrame):
        return [self.useForest(row) for _,row in ds.iterrows()]

#EOF