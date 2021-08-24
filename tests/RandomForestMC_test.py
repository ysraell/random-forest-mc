import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append("src/")


def version_test():
    from random_forest_mc import __version__

    assert __version__ == "0.2.0"


def test_LoadDicts():
    from random_forest_mc.utils import LoadDicts

    _ = LoadDicts("tests/")


def test_LoadDicts_content():
    from random_forest_mc.utils import LoadDicts

    dicts = LoadDicts("tests/")
    assert "datasets_metadata" in dicts.List


def test_RandomForestMC():
    from random_forest_mc.model import RandomForestMC

    _ = RandomForestMC()


def test_RandomForestMC_DatasetNotFound():
    from random_forest_mc.model import RandomForestMC, DatasetNotFound

    model = RandomForestMC()
    with pytest.raises(DatasetNotFound):
        model.fit()


def test_RandomForestMC_DatasetNotFound_Parallel():
    from random_forest_mc.model import RandomForestMC, DatasetNotFound

    model = RandomForestMC()
    with pytest.raises(DatasetNotFound):
        model.fitParallel()


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
    cls = RandomForestMC(target_col=params["target_col"])
    cls.fit(dataset)
    Pass = cls.Forest_size > 0
    cls.reset_forest()
    assert Pass and (cls.Forest_size == 0)


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
    cls = RandomForestMC(target_col=params["target_col"])
    cls.fitParallel(dataset=dataset, max_workers=4, thread_parallel_method=False)


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
    y_pred = cls.testForest(ds)
    accuracy_hard = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    y_pred = cls.testForest(ds, soft_voting=True)
    accuracy_soft = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    _ = cls.testForestProbs(ds)
    assert (accuracy_hard > 0.6) and (accuracy_soft > 0.6)


def test_RandomForestMC_fullCycle_titanic_Parallel_thread():
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
    cls.fitParallel(max_workers=4, thread_parallel_method=True)
    ds = dataset.sample(n=min(1000, dataset.shape[0]), random_state=51)
    y_test = ds[params["target_col"]].to_list()
    y_pred = cls.testForest(ds)
    accuracy_hard = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    y_pred = cls.testForest(ds, soft_voting=True)
    accuracy_soft = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    _ = cls.testForestProbs(ds)
    assert (accuracy_hard > 0.6) and (accuracy_soft > 0.6)


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
    cls.fitParallel(max_workers=4, thread_parallel_method=False)
    ds = dataset.sample(n=min(1000, dataset.shape[0]), random_state=51)
    y_test = ds[params["target_col"]].to_list()
    y_pred = cls.testForest(ds)
    accuracy_hard = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    y_pred = cls.testForest(ds, soft_voting=True)
    accuracy_soft = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    _ = cls.testForestProbs(ds)
    assert (accuracy_hard > 0.6) and (accuracy_soft > 0.6)


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
    y_pred = cls.testForest(ds)
    accuracy_hard = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    y_pred = cls.testForest(ds, soft_voting=True)
    accuracy_soft = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    _ = cls.testForestProbs(ds)
    assert (accuracy_hard > 0.3) and (accuracy_soft > 0.3)


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
    y_pred = cls.testForest(ds, soft_voting=True)
    _ = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    _ = cls.testForestProbs(ds)


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
    cls.fitParallel(max_workers=8, thread_parallel_method=False)
    ds = dataset.sample(n=min(1000, dataset.shape[0]), random_state=51)
    y_test = ds[params["target_col"]].to_list()
    y_pred = cls.testForest(ds)
    _ = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    y_pred = cls.testForest(ds, soft_voting=True)
    _ = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    _ = cls.testForestProbs(ds)
    assert cls.Forest_size == n_trees


# EOF
