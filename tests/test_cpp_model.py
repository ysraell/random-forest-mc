import numpy as np
import pandas as pd
import pytest
import pytest_check as check

import random_forest_mc as rf
from random_forest_mc.model import (
    RandomForestMC as PyRandomForestMC,
    DatasetNotFound,
    MissingValuesNotFound,
    DictValuesAllFeaturesMissing,
)
from random_forest_mc.cpp_model import (
    RandomForestMC as CppRandomForestMC,
    CPP_AVAILABLE,
)


def test_cpp_availability():
    assert CPP_AVAILABLE is True
    assert rf.CPP_AVAILABLE is True
    assert rf.RandomForestMCCPP is CppRandomForestMC
    assert rf.RandomForestMCPy is PyRandomForestMC


def test_engine_dispatch():
    m_auto = rf.RandomForestMC(n_trees=2, engine="auto")
    assert isinstance(m_auto, CppRandomForestMC)

    m_cpp = rf.RandomForestMC(n_trees=2, engine="cpp")
    assert isinstance(m_cpp, CppRandomForestMC)

    m_c = rf.RandomForestMC(n_trees=2, engine="c")
    assert isinstance(m_c, CppRandomForestMC)

    m_py = rf.RandomForestMC(n_trees=2, engine="python")
    assert isinstance(m_py, PyRandomForestMC)

    with pytest.raises(ValueError):
        rf.RandomForestMC(n_trees=2, engine="unknown_backend")


def test_cpp_dataset_not_found():
    model = CppRandomForestMC(n_trees=2)
    with pytest.raises(DatasetNotFound):
        model.fit()


def test_cpp_fit_predict_iris():
    df = pd.read_csv("datasets/iris.csv")
    model = rf.RandomForestMC(n_trees=8, target_col="variety", engine="cpp", random_seed=42)
    model.fit(df)

    assert len(model) == 8
    assert len(model.survived_scores) == 8
    assert all(score > 0.0 for score in model.survived_scores)

    # Predictions
    preds = model.predict(df)
    assert len(preds) == len(df)
    accuracy = np.mean(df["variety"].to_numpy() == np.array(preds))
    check.greater(accuracy, 0.85)

    # Probabilities
    probs = model.predict_proba(df)
    assert len(probs) == len(df)
    for row_p in probs:
        check.is_in("Setosa", row_p)
        check.is_in("Versicolor", row_p)
        check.is_in("Virginica", row_p)
        check.almost_equal(sum(row_p.values()), 1.0, abs=1e-5)


def test_cpp_feature_importance():
    df = pd.read_csv("datasets/iris.csv")
    model = rf.RandomForestMC(n_trees=8, target_col="variety", engine="cpp", random_seed=42)
    model.fit(df)

    imp = model.featImportance()
    assert isinstance(imp, dict)
    assert len(imp) == 4
    assert all(0.0 <= v <= 1.0 for v in imp.values())

    score_mean = model.featScoreMean()
    assert isinstance(score_mean, dict)
    assert len(score_mean) == 4

    pair_imp = model.featPairImportance()
    assert isinstance(pair_imp, dict)

    corr_df = model.featCorrDataFrame()
    assert isinstance(corr_df, pd.DataFrame)
    assert corr_df.shape == (4, 4)


def test_bidirectional_serialization_parity():
    df = pd.read_csv("datasets/iris.csv")

    # 1. C++ -> Python
    cpp_model = CppRandomForestMC(n_trees=8, target_col="variety", random_seed=42)
    cpp_model.fit(df)
    cpp_preds = cpp_model.predict(df)
    cpp_dict = cpp_model.model2dict()

    py_model = PyRandomForestMC(target_col="variety")
    py_model.dict2model(cpp_dict)
    py_preds = py_model.predict(df)

    assert cpp_preds == py_preds, "Pure Python model must produce exact predictions from C++ dict"

    # 2. Python -> C++
    py_model2 = PyRandomForestMC(n_trees=8, target_col="variety")
    py_model2.fit(df)
    py_preds2 = py_model2.predict(df)
    py_dict2 = py_model2.model2dict()

    cpp_model2 = CppRandomForestMC(target_col="variety")
    cpp_model2.dict2model(py_dict2)
    cpp_preds2 = cpp_model2.predict(df)

    assert py_preds2 == cpp_preds2, "C++ model must produce exact predictions from Python dict"


def test_cpp_missing_values_inference():
    df = pd.read_csv("datasets/iris.csv")
    model = rf.RandomForestMC(n_trees=8, target_col="variety", engine="cpp", random_seed=42)
    model.fit(df)

    # Row with NaN
    row_with_nan = df.iloc[0].copy()
    row_with_nan["petal.length"] = np.nan
    probs = model.predict_proba(row_with_nan)
    assert isinstance(probs, dict)
    check.almost_equal(sum(probs.values()), 1.0, abs=1e-5)

    # Matrix with NaN
    df_nan = df.copy()
    df_nan.loc[0:5, "petal.length"] = np.nan
    probs_batch = model.predict_proba(df_nan.head(6))
    assert len(probs_batch) == 6
    for p in probs_batch:
        check.almost_equal(sum(p.values()), 1.0, abs=1e-5)


def test_cpp_predictMissingValues():
    df = pd.read_csv("datasets/titanic.csv")[["Sex", "Age", "Pclass", "Survived"]].dropna().reset_index(drop=True)
    df["Pclass"] = df["Pclass"].astype(str)
    ds_cols = ["Sex", "Age", "Pclass"]
    target_col = "Survived"

    model = rf.RandomForestMC(target_col=target_col, engine="cpp", random_seed=42)
    model.fit(df)

    dict_values = {col: df[col].unique().tolist() for col in ds_cols}

    row = df.iloc[0].copy()
    row["Age"] = np.nan
    res = model.predictMissingValues(row, dict_values)
    assert isinstance(res, pd.DataFrame)
    assert len(res) > 0

    # Test error cases
    with pytest.raises(MissingValuesNotFound):
        model.predictMissingValues(df.iloc[0], dict_values)

    with pytest.raises(DictValuesAllFeaturesMissing):
        model.predictMissingValues(row, {"NonExistentCol": [1, 2]})


def test_cpp_voting_modes():
    df = pd.read_csv("datasets/iris.csv")

    for soft in (True, False):
        for weighted in (True, False):
            m = rf.RandomForestMC(n_trees=6, target_col="variety", engine="cpp", random_seed=42)
            m.setSoftVoting(soft)
            m.setWeightedTrees(weighted)
            m.fit(df)
            preds = m.predict(df.head(10))
            assert len(preds) == 10
            probs = m.predict_proba(df.head(10))
            assert len(probs) == 10
            for p in probs:
                assert abs(sum(p.values()) - 1.0) < 1e-4
