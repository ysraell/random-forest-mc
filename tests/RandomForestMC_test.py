import functools
import operator
import sys
from copy import deepcopy
from random import shuffle

import numpy as np
import pandas as pd
import pytest
import pytest_check as check
from json import JSONDecodeError


def flat(a):
    return functools.reduce(operator.iconcat, a, [])


sys.path.append("src/")
path_dict = "/tmp/datasets/model_dict.json"


def test_version():
    from random_forest_mc import __version__

    assert __version__ == "1.1.0"


# @pytest.mark.skip()
def test_LoadDicts():
    from random_forest_mc.utils import LoadDicts

    dicts = LoadDicts("tests/")
    _ = str(dicts)


# @pytest.mark.skip()
def test_LoadDicts_content():
    from random_forest_mc.utils import LoadDicts

    dicts = LoadDicts("tests/")
    assert "datasets_metadata" in dicts.List


# @pytest.mark.skip()
def test_LoadDicts_load_json_csv_like():
    from random_forest_mc.utils import LoadDicts

    _ = LoadDicts("../../../tmp/datasets/load_json_csv_like", ignore_errors=True)
    with pytest.raises(JSONDecodeError):
        _ = LoadDicts("../../../tmp/datasets/load_json_csv_like")


# @pytest.mark.skip()
def test_LoadDicts_load_json_empty():
    from random_forest_mc.utils import LoadDicts

    _ = LoadDicts("../../../tmp/datasets/load_json_empty", ignore_errors=True)
    with pytest.raises(JSONDecodeError):
        _ = LoadDicts("../../../tmp/datasets/load_json_empty")


# @pytest.mark.skip()
def test_LoadDicts_load_json_keyword():
    from random_forest_mc.utils import LoadDicts

    ld = LoadDicts("../../../tmp/datasets/load_json_keyword")
    assert len(ld.not_attr) > 0


# @pytest.mark.skip()
def test_RandomForestMC():
    from random_forest_mc.model import RandomForestMC

    cls = RandomForestMC()
    txt = "RandomForestMC(len(Forest)={},n_trees={},model_version={},module_version={})"
    assert cls.__repr__() == txt.format(
        len(cls.Forest), cls.n_trees, cls.model_version, cls.version
    )


# @pytest.mark.skip()
def test_RandomForestMC_DatasetNotFound():
    from random_forest_mc.model import RandomForestMC, DatasetNotFound

    model = RandomForestMC()
    with pytest.raises(DatasetNotFound):
        model.fit()


# @pytest.mark.skip()
def test_RandomForestMC_DatasetNotFound_Parallel():
    from random_forest_mc.model import RandomForestMC, DatasetNotFound

    model = RandomForestMC()
    with pytest.raises(DatasetNotFound):
        model.fitParallel()


# @pytest.mark.skip()
def test_RandomForestMC_process_dataset():
    from random_forest_mc.model import RandomForestMC
    from random_forest_mc.utils import LoadDicts

    dicts = LoadDicts("tests/")
    dataset_dict = dicts.datasets_metadata
    ds_name = "titanic"
    params = dataset_dict[ds_name]
    dataset = pd.read_csv(params["csv_path"])
    cls = RandomForestMC(target_col=params["target_col"])
    cls.process_dataset(dataset)


# @pytest.mark.skip()
def test_RandomForestMC_fit():
    from random_forest_mc.model import RandomForestMC
    from random_forest_mc.utils import LoadDicts

    dicts = LoadDicts("tests/")
    dataset_dict = dicts.datasets_metadata
    ds_name = "titanic"
    params = dataset_dict[ds_name]
    dataset = (
        pd.read_csv(params["csv_path"])[params["ds_cols"] + [params["target_col"]]]
        .dropna()
        .reset_index(drop=True)
    )
    dataset["Age"] = dataset["Age"].astype(np.uint8)
    dataset["SibSp"] = dataset["SibSp"].astype(np.uint8)
    dataset["Pclass"] = dataset["Pclass"].astype(str)
    dataset["Fare"] = dataset["Fare"].astype(np.uint32)
    cls = RandomForestMC(
        target_col=params["target_col"],
        max_discard_trees=20,
        th_decease_verbose=True,
        temporal_features=True,
    )
    cls.process_dataset(dataset)
    check.is_false(cls.temporal_features)
    dataset.insert(len(dataset.columns), "coluna_vazia", "None")
    columns = {
        col: f"{col}_{i}"
        for i, col in enumerate(dataset.columns)
        if col != params["target_col"]
    }
    dataset = dataset.rename(columns=columns)
    cls = RandomForestMC(
        target_col=params["target_col"],
        max_discard_trees=20,
        th_decease_verbose=True,
        temporal_features=True,
    )
    cls.process_dataset(dataset)
    check.is_true(cls.temporal_features)
    cls.fit(dataset)

    Tree = cls.data[0]
    check.is_in("DecisionTreeMC(", repr(Tree))

    check.equal(Tree, Tree)
    with check.raises(TypeError):
        Tree == Tree.data

    _ = Tree > Tree
    with check.raises(TypeError):
        Tree > Tree.data

    _ = Tree >= Tree
    with check.raises(TypeError):
        Tree >= Tree.data


# @pytest.mark.skip()
def test_RandomForestMC_fitParallel():
    from random_forest_mc.model import RandomForestMC
    from random_forest_mc.utils import LoadDicts

    dicts = LoadDicts("tests/")
    dataset_dict = dicts.datasets_metadata
    ds_name = "titanic"
    params = dataset_dict[ds_name]
    dataset = (
        pd.read_csv(params["csv_path"])[params["ds_cols"] + [params["target_col"]]]
        .dropna()
        .reset_index(drop=True)
    )
    dataset["Age"] = dataset["Age"].astype(np.uint8)
    dataset["SibSp"] = dataset["SibSp"].astype(np.uint8)
    dataset["Pclass"] = dataset["Pclass"].astype(str)
    dataset["Fare"] = dataset["Fare"].astype(np.uint32)
    dataset.insert(len(dataset.columns), "coluna_vazia", "None")
    cls = RandomForestMC(
        target_col=params["target_col"], max_discard_trees=20, th_decease_verbose=True
    )
    cls.fitParallel(dataset=dataset, max_workers=4)


# @pytest.mark.skip()
def test_RandomForestMC_fit_max_depth():
    from random_forest_mc.model import RandomForestMC
    from random_forest_mc.utils import LoadDicts

    dicts = LoadDicts("tests/")
    dataset_dict = dicts.datasets_metadata
    ds_name = "titanic"
    params = dataset_dict[ds_name]
    dataset = (
        pd.read_csv(params["csv_path"])[params["ds_cols"] + [params["target_col"]]]
        .dropna()
        .reset_index(drop=True)
    )
    dataset["Age"] = dataset["Age"].astype(np.uint8)
    dataset["SibSp"] = dataset["SibSp"].astype(np.uint8)
    dataset["Pclass"] = dataset["Pclass"].astype(str)
    dataset["Fare"] = dataset["Fare"].astype(np.uint32)
    dataset.insert(len(dataset.columns), "coluna_vazia", "None")
    max_depth = 2
    cls = RandomForestMC(
        target_col=params["target_col"], max_discard_trees=20, th_decease_verbose=True
    )
    cls.fitParallel(dataset=dataset, max_workers=4)
    max_depth_got = max(flat(cls.trees2depths))
    check.greater(max_depth_got, max_depth)
    cls = RandomForestMC(
        target_col=params["target_col"],
        max_discard_trees=20,
        th_decease_verbose=True,
        max_depth=2,
    )
    cls.fitParallel(dataset=dataset, max_workers=4)
    max_depth_got = max(flat(cls.trees2depths))
    check.equal(max_depth_got, max_depth)


# @pytest.mark.skip()
def test_RandomForestMC_fit_min_samples_split():
    from random_forest_mc.model import RandomForestMC
    from random_forest_mc.utils import LoadDicts

    dicts = LoadDicts("tests/")
    dataset_dict = dicts.datasets_metadata
    ds_name = "titanic"
    params = dataset_dict[ds_name]
    dataset = (
        pd.read_csv(params["csv_path"])[params["ds_cols"] + [params["target_col"]]]
        .dropna()
        .reset_index(drop=True)
    )
    dataset["Age"] = dataset["Age"].astype(np.uint8)
    dataset["SibSp"] = dataset["SibSp"].astype(np.uint8)
    dataset["Pclass"] = dataset["Pclass"].astype(str)
    dataset["Fare"] = dataset["Fare"].astype(np.uint32)
    dataset.insert(len(dataset.columns), "coluna_vazia", "None")
    cls = RandomForestMC(
        target_col=params["target_col"], max_discard_trees=20, th_decease_verbose=True
    )
    cls.fitParallel(dataset=dataset, max_workers=4)
    max_depth_min_samples_split_1 = max(flat(cls.trees2depths))
    cls = RandomForestMC(
        target_col=params["target_col"],
        max_discard_trees=20,
        th_decease_verbose=True,
        min_samples_split=10,
    )
    cls.fitParallel(dataset=dataset, max_workers=4)
    max_depth_min_samples_split_10 = max(flat(cls.trees2depths))
    check.less(max_depth_min_samples_split_10, max_depth_min_samples_split_1)


# @pytest.mark.skip()
def test_RandomForestMC_fitParallel_featImportance():
    from random_forest_mc.model import RandomForestMC
    from random_forest_mc.utils import LoadDicts

    dicts = LoadDicts("tests/")
    dataset_dict = dicts.datasets_metadata
    ds_name = "titanic"
    params = dataset_dict[ds_name]
    dataset = (
        pd.read_csv(params["csv_path"])[params["ds_cols"] + [params["target_col"]]]
        .dropna()
        .reset_index(drop=True)
    )
    dataset["Age"] = dataset["Age"].astype(np.uint8)
    dataset["SibSp"] = dataset["SibSp"].astype(np.uint8)
    dataset["Pclass"] = dataset["Pclass"].astype(str)
    dataset["Fare"] = dataset["Fare"].astype(np.uint32)
    cls = RandomForestMC(target_col=params["target_col"], max_discard_trees=8)
    cls.fitParallel(dataset=dataset, max_workers=4)
    featCount_stats, featCount_list = cls.featCount()
    featImportance = cls.featImportance()
    featScoreMean = cls.featScoreMean()
    featPairImportance = cls.featPairImportance()
    featCorrDataFrame = cls.featCorrDataFrame()
    check.is_instance(featCount_stats, tuple)
    check.is_instance(featCount_list, list)
    check.is_instance(featImportance, dict)
    check.is_instance(featScoreMean, dict)
    check.is_instance(featPairImportance, dict)
    check.is_instance(featCorrDataFrame, pd.DataFrame)
    check.is_true(all([isinstance(val, float) for val in featCount_stats[:2]]))
    check.is_true(all([isinstance(val, int) for val in featCount_stats[2:]]))
    check.is_true(all([isinstance(count, int) for count in featCount_list]))
    for feat, count in featImportance.items():
        check.is_true(
            all([isinstance(feat, str), isinstance(count, float), count <= 1])
        )
    for feat, count in featScoreMean.items():
        check.is_true(
            all([isinstance(feat, str), isinstance(count, float), count <= 1])
        )
    for pair, count in featPairImportance.items():
        check.is_true(
            all(
                [
                    isinstance(pair, tuple),
                    isinstance(pair[0], str),
                    isinstance(pair[1], str),
                    isinstance(count, float),
                    count <= 1,
                ]
            )
        )


# @pytest.mark.skip()
def test_RandomForestMC_fitParallel_sampleClassFeatImportance():
    from random_forest_mc.model import RandomForestMC
    from random_forest_mc.utils import LoadDicts

    dicts = LoadDicts("tests/")
    dataset_dict = dicts.datasets_metadata
    ds_name = "titanic"
    params = dataset_dict[ds_name]
    target_col = params["target_col"]
    dataset = (
        pd.read_csv(params["csv_path"])[params["ds_cols"] + [params["target_col"]]]
        .dropna()
        .reset_index(drop=True)
    )
    dataset["Age"] = dataset["Age"].astype(np.uint8)
    dataset["SibSp"] = dataset["SibSp"].astype(np.uint8)
    dataset["Pclass"] = dataset["Pclass"].astype(str)
    dataset["Fare"] = dataset["Fare"].astype(np.uint32)
    cls = RandomForestMC(target_col=target_col, max_discard_trees=8)
    cls.fitParallel(dataset=dataset, max_workers=4)
    for row, Class in [
        (dataset.query(f'{target_col} == "1"').reset_index(drop=True).loc[0], "1"),
        (dataset.query(f'{target_col} == "0"').reset_index(drop=True).loc[0], "0"),
    ]:
        featCount_stats, featCount_list = cls.sampleClassFeatCount(row, Class)
        featImportance = cls.sampleClassFeatImportance(row, Class)
        featScoreMean = cls.sampleClassFeatScoreMean(row, Class)
        featPairImportance = cls.sampleClassFeatPairImportance(row, Class)
        featCorrDataFrame = cls.sampleClassFeatCorrDataFrame(row, Class)
        check.is_instance(featCount_stats, tuple)
        check.is_instance(featCount_list, list)
        check.is_instance(featImportance, dict)
        check.is_instance(featScoreMean, dict)
        check.is_instance(featPairImportance, dict)
        check.is_instance(featCorrDataFrame, pd.DataFrame)
        check.is_true(all([isinstance(val, float) for val in featCount_stats[:2]]))
        check.is_true(all([isinstance(val, int) for val in featCount_stats[2:]]))
        check.is_true(all([isinstance(count, int) for count in featCount_list]))
        for feat, count in featImportance.items():
            check.is_true(
                all([isinstance(feat, str), isinstance(count, float), count <= 1])
            )
        for feat, count in featScoreMean.items():
            check.is_true(
                all([isinstance(feat, str), isinstance(count, float), count <= 1])
            )
        for pair, count in featPairImportance.items():
            check.is_true(
                all(
                    [
                        isinstance(pair, tuple),
                        isinstance(pair[0], str),
                        isinstance(pair[1], str),
                        isinstance(count, float),
                        count <= 1,
                    ]
                )
            )


# @pytest.mark.skip()
def test_RandomForestMC_fit_get_best_tree_False():
    from random_forest_mc.model import RandomForestMC
    from random_forest_mc.utils import LoadDicts

    dicts = LoadDicts("tests/")
    dataset_dict = dicts.datasets_metadata
    ds_name = "titanic"
    params = dataset_dict[ds_name]
    dataset = (
        pd.read_csv(params["csv_path"])[params["ds_cols"] + [params["target_col"]]]
        .dropna()
        .reset_index(drop=True)
    )
    dataset["Age"] = dataset["Age"].astype(np.uint8)
    dataset["SibSp"] = dataset["SibSp"].astype(np.uint8)
    dataset["Pclass"] = dataset["Pclass"].astype(str)
    dataset["Fare"] = dataset["Fare"].astype(np.uint32)
    cls = RandomForestMC(
        target_col=params["target_col"],
        get_best_tree=False,
        max_discard_trees=20,
        th_decease_verbose=True,
    )
    cls.fit(dataset)


# @pytest.mark.skip()
def test_RandomForestMC_fitParallel_get_best_tree_False():
    from random_forest_mc.model import RandomForestMC
    from random_forest_mc.utils import LoadDicts

    dicts = LoadDicts("tests/")
    dataset_dict = dicts.datasets_metadata
    ds_name = "titanic"
    params = dataset_dict[ds_name]
    dataset = (
        pd.read_csv(params["csv_path"])[params["ds_cols"] + [params["target_col"]]]
        .dropna()
        .reset_index(drop=True)
    )
    dataset["Age"] = dataset["Age"].astype(np.uint8)
    dataset["SibSp"] = dataset["SibSp"].astype(np.uint8)
    dataset["Pclass"] = dataset["Pclass"].astype(str)
    dataset["Fare"] = dataset["Fare"].astype(np.uint32)
    cls = RandomForestMC(
        target_col=params["target_col"],
        get_best_tree=False,
        max_discard_trees=20,
        th_decease_verbose=True,
    )
    cls.fitParallel(dataset=dataset, max_workers=4)


# @pytest.mark.skip()
def test_RandomForestMC_save_load_model():
    from random_forest_mc.model import RandomForestMC
    from random_forest_mc.utils import LoadDicts, dump_file_json, load_file_json

    dicts = LoadDicts("tests/")
    dataset_dict = dicts.datasets_metadata
    ds_name = "titanic"
    params = dataset_dict[ds_name]
    dataset = (
        pd.read_csv(params["csv_path"])[params["ds_cols"] + [params["target_col"]]]
        .dropna()
        .reset_index(drop=True)
    )
    dataset["Age"] = dataset["Age"].astype(np.uint8)
    dataset["SibSp"] = dataset["SibSp"].astype(np.uint8)
    dataset["Pclass"] = dataset["Pclass"].astype(str)
    dataset["Fare"] = dataset["Fare"].astype(np.uint32)
    cls = RandomForestMC(target_col=params["target_col"])
    cls.fit(dataset)
    Forest_size = cls.Forest_size
    sum_survived_scores = sum(cls.survived_scores)
    ModelDict = cls.model2dict()
    dump_file_json(path_dict, ModelDict)
    del ModelDict
    ModelDict = load_file_json(path_dict)
    cls = RandomForestMC(target_col=params["target_col"])
    cls.process_dataset(dataset)
    cls.dict2model(ModelDict)
    check.equal(cls.Forest_size, Forest_size)
    check.almost_equal(sum(cls.survived_scores), sum_survived_scores)


# @pytest.mark.skip()
def test_RandomForestMC_predictl():
    from random_forest_mc.model import RandomForestMC
    from random_forest_mc.utils import LoadDicts

    dicts = LoadDicts("tests/")
    dataset_dict = dicts.datasets_metadata
    ds_name = "titanic"
    params = dataset_dict[ds_name]
    dataset = (
        pd.read_csv(params["csv_path"])[params["ds_cols"] + [params["target_col"]]]
        .dropna()
        .reset_index(drop=True)
    )
    dataset["Age"] = dataset["Age"].astype(np.uint8)
    dataset["SibSp"] = dataset["SibSp"].astype(np.uint8)
    dataset["Pclass"] = dataset["Pclass"].astype(str)
    dataset["Fare"] = dataset["Fare"].astype(np.uint32)
    cls = RandomForestMC(target_col=params["target_col"])
    cls.fit(dataset)
    row = dataset.reset_index(drop=True).loc[0]

    predict_row = cls.predict(row)
    check.is_instance(predict_row, dict)

    predict_ds = cls.predict(dataset.sample(n=10))
    check.is_instance(predict_ds, list)

    predict_probs_ds = cls.predict_proba(dataset.sample(n=10))
    check.is_instance(predict_probs_ds, list)
    for leaf in predict_probs_ds:
        check.is_instance(leaf, dict)

    with check.raises(TypeError):
        _ = cls.predict(row.array)


# @pytest.mark.skip()
def test_RandomForestMC_mergeForest_dropduplicated():
    from random_forest_mc.model import RandomForestMC
    from random_forest_mc.utils import LoadDicts

    dicts = LoadDicts("tests/")
    dataset_dict = dicts.datasets_metadata
    ds_name = "titanic"
    params = dataset_dict[ds_name]
    dataset = (
        pd.read_csv(params["csv_path"])[params["ds_cols"] + [params["target_col"]]]
        .dropna()
        .reset_index(drop=True)
    )
    dataset["Age"] = dataset["Age"].astype(np.uint8)
    dataset["SibSp"] = dataset["SibSp"].astype(np.uint8)
    dataset["Pclass"] = dataset["Pclass"].astype(str)
    dataset["Fare"] = dataset["Fare"].astype(np.uint32)
    cls = RandomForestMC(target_col=params["target_col"])
    cls.fit(dataset)
    Forest_size = cls.Forest_size
    cls.mergeForest(cls)
    cls.drop_duplicated_trees()
    check.equal(cls.Forest_size, Forest_size)
    cls.mergeForest(cls)
    check.equal(cls.Forest_size, 2 * Forest_size)
    cls.mergeForest(cls, 11, "random")
    check.equal(cls.Forest_size, 11)
    cls.mergeForest(cls, 8, "score")
    check.equal(cls.Forest_size, 8)

    with check.raises(TypeError):
        cls.mergeForest(cls.data)

    cls_other = deepcopy(cls)
    check.equal(cls_other, cls)
    with check.raises(ValueError):
        cls_other.feature_cols.pop(0)
        cls.mergeForest(cls_other)

    cls_other = deepcopy(cls)
    check.equal(cls_other, cls)
    with check.raises(ValueError):
        cls_other.class_vals.append("3")
        cls.mergeForest(cls_other)

    with check.raises(TypeError):
        cls == cls_other.data

    for Tree in cls:
        for Tree_other in cls_other:
            _ = Tree <= Tree_other
            _ = Tree < Tree_other
            _ = Tree >= Tree_other
            _ = Tree > Tree_other


# @pytest.mark.skip()
def test_RandomForestMC_fullCycle_titanic():
    from random_forest_mc.model import RandomForestMC
    from random_forest_mc.utils import LoadDicts

    dicts = LoadDicts("tests/")
    dataset_dict = dicts.datasets_metadata
    ds_name = "titanic"
    params = dataset_dict[ds_name]
    dataset = (
        pd.read_csv(params["csv_path"])[params["ds_cols"] + [params["target_col"]]]
        .dropna()
        .reset_index(drop=True)
    )
    dataset["Age"] = dataset["Age"].astype(np.uint8)
    dataset["SibSp"] = dataset["SibSp"].astype(np.uint8)
    dataset["Pclass"] = dataset["Pclass"].astype(str)
    dataset["Fare"] = dataset["Fare"].astype(np.uint32)
    cls = RandomForestMC(
        n_trees=32, target_col=params["target_col"], max_discard_trees=16
    )
    cls.process_dataset(dataset)
    cls.fit()
    ds = dataset.sample(n=min(1000, dataset.shape[0]), random_state=51)
    y_test = ds[params["target_col"]].to_list()
    cls.setSoftVoting(False)
    cls.setWeightedTrees(False)
    y_pred = cls.testForest(ds)
    accuracy_hard = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    cls.setSoftVoting(True)
    cls.setWeightedTrees(False)
    y_pred = cls.testForest(ds)
    accuracy_soft = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    cls.setSoftVoting(False)
    cls.setWeightedTrees(True)
    y_pred = cls.testForest(ds)
    accuracy_hard_weighted = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    cls.setSoftVoting(True)
    cls.setWeightedTrees(True)
    y_pred = cls.testForest(ds)
    accuracy_soft_weighted = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    _ = cls.testForestProbs(ds)
    check.greater(accuracy_hard, 0.6)
    check.greater(accuracy_soft, 0.6)
    check.greater(accuracy_hard_weighted, 0.6)
    check.greater(accuracy_soft_weighted, 0.6)


# @pytest.mark.skip()
def test_RandomForestMC_fullCycle_titanic_Parallel_process():
    from random_forest_mc.model import RandomForestMC
    from random_forest_mc.utils import LoadDicts

    dicts = LoadDicts("tests/")
    dataset_dict = dicts.datasets_metadata
    ds_name = "titanic"
    params = dataset_dict[ds_name]
    dataset = (
        pd.read_csv(params["csv_path"])[params["ds_cols"] + [params["target_col"]]]
        .dropna()
        .reset_index(drop=True)
    )
    dataset["Age"] = dataset["Age"].astype(np.uint8)
    dataset["SibSp"] = dataset["SibSp"].astype(np.uint8)
    dataset["Pclass"] = dataset["Pclass"].astype(str)
    dataset["Fare"] = dataset["Fare"].astype(np.uint32)
    cls = RandomForestMC(
        n_trees=32, target_col=params["target_col"], max_discard_trees=16
    )
    cls.process_dataset(dataset)
    cls.fitParallel(max_workers=4)
    ds = dataset.sample(n=min(1000, dataset.shape[0]), random_state=51)
    y_test = ds[params["target_col"]].to_list()
    cls.setSoftVoting(False)
    cls.setWeightedTrees(False)
    y_pred = cls.testForest(ds)
    accuracy_hard = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    cls.setSoftVoting(True)
    cls.setWeightedTrees(False)
    y_pred = cls.testForest(ds)
    accuracy_soft = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    cls.setSoftVoting(False)
    cls.setWeightedTrees(True)
    y_pred = cls.testForest(ds)
    accuracy_hard_weighted = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    cls.setSoftVoting(True)
    cls.setWeightedTrees(True)
    y_pred = cls.testForest(ds)
    accuracy_soft_weighted = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    _ = cls.testForestProbs(ds)
    check.greater(accuracy_hard, 0.6)
    check.greater(accuracy_soft, 0.6)
    check.greater(accuracy_hard_weighted, 0.6)
    check.greater(accuracy_soft_weighted, 0.6)


# @pytest.mark.skip()
def test_RandomForestMC_fullCycle_iris():
    from random_forest_mc.model import RandomForestMC
    from random_forest_mc.utils import LoadDicts

    dicts = LoadDicts("tests/")
    dataset_dict = dicts.datasets_metadata
    ds_name = "iris"
    params = dataset_dict[ds_name]
    dataset = (
        pd.read_csv(params["csv_path"])[params["ds_cols"] + [params["target_col"]]]
        .dropna()
        .reset_index(drop=True)
    )
    dataset.rename(
        columns={col: col.replace(".", "_") for col in dataset.columns}, inplace=True
    )
    params["ds_cols"] = [col.replace(".", "_") for col in params["ds_cols"]]
    cls = RandomForestMC(
        n_trees=8, target_col=params["target_col"], max_discard_trees=4
    )
    cls.process_dataset(dataset)
    cls.fit()
    ds = dataset.sample(n=min(1000, dataset.shape[0]), random_state=51)
    y_test = ds[params["target_col"]].to_list()
    cls.setSoftVoting(False)
    cls.setWeightedTrees(False)
    y_pred = cls.testForest(ds)
    accuracy_hard = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    cls.setSoftVoting(True)
    cls.setWeightedTrees(False)
    y_pred = cls.testForest(ds)
    accuracy_soft = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    cls.setSoftVoting(False)
    cls.setWeightedTrees(True)
    y_pred = cls.testForest(ds)
    accuracy_hard_weighted = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    cls.setSoftVoting(True)
    cls.setWeightedTrees(True)
    y_pred = cls.testForest(ds)
    accuracy_soft_weighted = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    _ = cls.testForestProbs(ds)
    check.greater(accuracy_hard, 0.6)
    check.greater(accuracy_soft, 0.6)
    check.greater(accuracy_hard_weighted, 0.6)
    check.greater(accuracy_soft_weighted, 0.6)


# @pytest.mark.skip()
def test_RandomForestMC_fullCycle_creditcard():
    from random_forest_mc.model import RandomForestMC
    from random_forest_mc.utils import LoadDicts

    dicts = LoadDicts("tests/")
    dataset_dict = dicts.datasets_metadata
    ds_name = "creditcard_trans_float"
    params = dataset_dict[ds_name]
    dataset = (
        pd.read_csv(params["csv_path"])[params["ds_cols"] + [params["target_col"]]]
        .dropna()
        .reset_index(drop=True)
    )
    cls = RandomForestMC(
        n_trees=32, target_col=params["target_col"], max_discard_trees=16
    )
    cls.process_dataset(dataset)
    cls.fit()
    ds = dataset.sample(n=min(1000, dataset.shape[0]), random_state=51)
    y_test = ds[params["target_col"]].to_list()
    y_pred = cls.testForest(ds)
    _ = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    _ = cls.testForestProbs(ds)


# @pytest.mark.skip()
def test_RandomForestMC_fullCycle_creditcard_missing_values():
    from random_forest_mc.model import RandomForestMC
    from random_forest_mc.utils import LoadDicts

    dicts = LoadDicts("tests/")
    dataset_dict = dicts.datasets_metadata
    ds_name = "creditcard_trans_float"
    params = dataset_dict[ds_name]
    dataset = (
        pd.read_csv(params["csv_path"])[params["ds_cols"] + [params["target_col"]]]
        .dropna()
        .reset_index(drop=True)
    )
    cls = RandomForestMC(
        n_trees=32, target_col=params["target_col"], max_discard_trees=16
    )
    cls.process_dataset(dataset)
    cls.fit()
    cols = list(dataset.columns)
    _ = cols.pop(cols.index("Class"))
    shuffle(cols)
    dataset = dataset[cols[:10] + ["Class"]]
    row = dataset.reset_index(drop=True).loc[0]

    predict_row = cls.predict(row)
    check.is_instance(predict_row, dict)

    predict_ds = cls.predict(dataset.sample(n=10))
    check.is_instance(predict_ds, list)

    predict_probs_ds = cls.predict_proba(dataset.sample(n=10))
    check.is_instance(predict_probs_ds, list)
    for leaf in predict_probs_ds:
        check.is_instance(leaf, dict)


# @pytest.mark.skip()
def test_RandomForestMC_fullCycle_creditcard_Parallel_process():
    from random_forest_mc.model import RandomForestMC
    from random_forest_mc.utils import LoadDicts

    dicts = LoadDicts("tests/")
    dataset_dict = dicts.datasets_metadata
    ds_name = "creditcard_trans_float"
    params = dataset_dict[ds_name]
    dataset = (
        pd.read_csv(params["csv_path"])[params["ds_cols"] + [params["target_col"]]]
        .dropna()
        .reset_index(drop=True)
    )
    n_trees = 32
    cls = RandomForestMC(
        n_trees=n_trees, target_col=params["target_col"], max_discard_trees=16
    )
    cls.process_dataset(dataset)
    cls.fitParallel(max_workers=8)
    ds = dataset.sample(n=min(1000, dataset.shape[0]), random_state=51)
    y_test = ds[params["target_col"]].to_list()
    y_pred = cls.testForest(ds)
    _ = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    _ = cls.testForestProbs(ds)
    check.equal(cls.Forest_size, n_trees)


# @pytest.mark.skip()
def test_RandomForestMC_predictMissingValues():
    from random_forest_mc.model import RandomForestMC, MissingValuesNotFound
    from random_forest_mc.utils import LoadDicts

    # Load basics:
    dicts = LoadDicts("../../random-forest-mc/tests/")
    dataset_dict = dicts.datasets_metadata
    ds_name = "titanic"
    params = dataset_dict[ds_name]
    dataset = (
        pd.read_csv(params["csv_path"])[params["ds_cols"] + [params["target_col"]]]
        .dropna()
        .reset_index(drop=True)
    )
    dataset["Age"] = dataset["Age"].astype(np.uint8)
    dataset["SibSp"] = dataset["SibSp"].astype(np.uint8)
    dataset["Pclass"] = dataset["Pclass"].astype(str)
    dataset["Fare"] = dataset["Fare"].astype(np.uint32)
    ds_cols = params["ds_cols"]
    target_col = params["target_col"]
    cls = RandomForestMC(target_col=target_col)
    cls.fit(dataset)
    
    # Create some missing data
    df_tmp = dataset.sample(frac=0.2).reset_index(drop=True)
    mask_random = np.random.choice([True, False], size=df_tmp[ds_cols].shape, p=[0.7, 0.3])
    dataset_missing_values = df_tmp[ds_cols].mask(~mask_random)
    dataset_missing_values[target_col] = df_tmp[target_col]
    
    dict_values = {col: dataset[col].unique().tolist() for col in ds_cols}
    
    for i, row in dataset_missing_values.iterrows():
        if row.isna().any():
            break
    df_tmp = cls.predictMissingValues(row, dict_values)
    check.is_instance(df_tmp, pd.DataFrame)
    check.greater(len(df_tmp), 0)
    
    while True:
        df_tmp = dataset_missing_values.sample(n=20)
        if df_tmp.isna().any().any():
            break
    df_tmp = cls.predictMissingValues(df_tmp, dict_values)
    check.is_instance(df_tmp, pd.DataFrame)
    check.greater(len(df_tmp), 0)
    
    with pytest.raises(MissingValuesNotFound):
        _ = cls.predictMissingValues(dataset.sample(n=20), dict_values)
    


# EOF
