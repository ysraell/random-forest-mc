import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append("src/")


def version_test():
    from random_forest_mc import __version__

    assert __version__ == "0.1.0"


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
        n_trees=8, target_col=params["target_col"], max_discard_trees=4
    )
    cls.process_dataset(dataset)
    cls.fit()
    y_test = dataset[params["target_col"]].to_list()
    y_pred = cls.testForest(dataset)
    accuracy_hard = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    y_pred = cls.testForest(dataset, soft_voting=True)
    accuracy_soft = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    _ = cls.testForestProbs(dataset)
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
    y_test = dataset[params["target_col"]].to_list()
    y_pred = cls.testForest(dataset)
    accuracy_hard = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    y_pred = cls.testForest(dataset, soft_voting=True)
    accuracy_soft = sum([v == p for v, p in zip(y_test, y_pred)]) / len(y_pred)
    _ = cls.testForestProbs(dataset)
    assert (accuracy_hard > 0.3) and (accuracy_soft > 0.3)


# EOF
