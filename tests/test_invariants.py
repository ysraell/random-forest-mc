"""Engine-independent invariants of the v1.6.0 method changes (Python and C++ engines)."""

import json
import logging

import numpy as np
import pandas as pd
import pytest

from random_forest_mc.cpp_model import CPP_AVAILABLE
from random_forest_mc.cpp_model import RandomForestMC as CppRandomForestMC
from random_forest_mc.model import RandomForestMC as PyRandomForestMC
from random_forest_mc.selection import ensemble_select, mean_pairwise_agreement
from random_forest_mc.tree import DecisionTreeMC, route_matrix, walk_leaves

ENGINES = ["python", "cpp"] if CPP_AVAILABLE else ["python"]


def make_model(engine, **kwargs):
    if engine == "cpp":
        return CppRandomForestMC(**kwargs)
    kwargs.pop("random_seed", None)
    return PyRandomForestMC(**kwargs)


def fit(model, df):
    if isinstance(model, PyRandomForestMC):
        model.fit(df.copy(), disable_progress_bar=True)
    else:
        model.fit(df.copy())
    return model


def ties_dataset(seed=0, n_per_class=30):
    """Balanced binary data with heavily tied numeric features and a categorical feature."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_class
    y = np.repeat(["a", "b"], n_per_class)
    return pd.DataFrame(
        {
            "x1": rng.integers(0, 4, n) + (y == "b"),
            "x2": rng.integers(0, 3, n),
            "x3": np.round(rng.normal(size=n), 1),
            "c1": rng.choice(["u", "v", "w"], n),
            "target": y,
        }
    )


def noise_dataset(seed=0, n=400, d=6):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    df["target"] = rng.choice(["0", "1"], n)
    return df


def synthetic_informative(seed=0, n=1200, n_noise=10):
    rng = np.random.default_rng(seed)
    informative = rng.normal(size=(n, 3))
    logit = 2.5 * informative[:, 0] - 2.0 * informative[:, 1] + 1.5 * informative[:, 2]
    y = (logit + rng.normal(scale=0.5, size=n) > 0).astype(int).astype(str)
    df = pd.DataFrame(informative, columns=["inf0", "inf1", "inf2"])
    df["red0"] = informative[:, 0] + rng.normal(scale=0.3, size=n)
    df["red1"] = informative[:, 1] + rng.normal(scale=0.3, size=n)
    for i in range(n_noise):
        df[f"noise{i}"] = rng.normal(size=n)
    df["target"] = y
    return df


def model_dict_from_tree(data, class_vals, feature_cols, numeric_cols):
    return {
        "min_feature": 2,
        "max_feature": len(feature_cols),
        "n_trees": 1,
        "class_vals": class_vals,
        "survived_scores": [1.0],
        "version": "1.6.0",
        "numeric_cols": numeric_cols,
        "feature_cols": feature_cols,
        "type_of_cols": {f: ("numeric" if f in numeric_cols else "categorical") for f in feature_cols},
        "target_col": "target",
        "tree_format": 2,
        "Forest": [
            {
                "data": data,
                "class_vals": class_vals,
                "survived_score": 1.0,
                "features": feature_cols,
                "used_features": feature_cols,
                "tree_format": 2,
            }
        ],
    }


# --- Item 1: thresholds reproduce the training partition ---


def leaf_partition_violations(tree: DecisionTreeMC, df: pd.DataFrame, target_col: str) -> int:
    """Leaves whose training count or class distribution differs from what routing the
    training rows produces, plus unreachable leaves."""
    columns = {col: df[col].to_numpy() for col in df.columns}
    routed = {id(leaf): idx for leaf, idx in route_matrix(tree.data, columns, np.arange(len(df)))}
    violations = 0
    y = df[target_col].astype(str).to_numpy()
    for leaf in walk_leaves(tree.data):
        idx = routed.get(id(leaf), np.array([], dtype=int))
        if len(idx) != leaf["n"]:
            violations += 1
            continue
        for c, p in leaf["leaf"].items():
            if not np.isclose(np.mean(y[idx] == str(c)), p):
                violations += 1
                break
    return violations


SPLIT_CONFIGS = [
    {},
    {"split_supervision": 1},
    {"split_supervision": 2},
    {"categorical_split": "ordered"},
    {"categorical_split": "ordered", "split_supervision": 2},
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("config", SPLIT_CONFIGS)
def test_training_rows_reach_their_leaves(engine, config):
    df = ties_dataset(seed=1)
    # Training batch = whole dataset (balanced classes, no validation rows).
    model = make_model(
        engine, n_trees=12, target_col="target", batch_train_pclass=1000, batch_val_pclass=0, random_seed=3, **config
    )
    fit(model, df)
    for tree in model.data:
        assert leaf_partition_violations(tree, df, "target") == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_legacy_split_rule_can_break_partition(engine):
    """Documents the v1.5.0 bug fixed by the midpoint rule (tie fallback / two-sample case)."""
    total = 0
    for seed in range(5):
        df = ties_dataset(seed=seed)
        np.random.seed(seed)
        model = make_model(
            engine, n_trees=12, target_col="target", batch_train_pclass=1000, batch_val_pclass=0,
            random_seed=seed, legacy_v150=True,
        )
        fit(model, df)
        total += sum(leaf_partition_violations(tree, df, "target") for tree in model.data)
    assert total > 0


# --- Item 2: missing-value marginalization ---

ASYM_TREE = {
    "f1": {
        "split": {
            "feat_type": "numeric",
            "split_val": 0.5,
            "n_ge": 6,
            "n_lt": 2,
            ">=": {"leaf": {"A": 1.0}, "depth": "2#", "n": 6},
            "<": {
                "f2": {
                    "split": {
                        "feat_type": "numeric",
                        "split_val": 0.5,
                        "n_ge": 1,
                        "n_lt": 1,
                        ">=": {"leaf": {"B": 1.0}, "depth": "3#", "n": 1},
                        "<": {"leaf": {"C": 1.0}, "depth": "3#", "n": 1},
                    }
                }
            },
        }
    }
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "strategy, expected",
    [("weighted", {"A": 0.75, "B": 0.125, "C": 0.125}), ("uniform_leaves", {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})],
)
def test_missing_value_marginalization(engine, strategy, expected):
    model = make_model(engine, target_col="target", missing_strategy=strategy)
    model.dict2model(model_dict_from_tree(ASYM_TREE, ["A", "B", "C"], ["f1", "f2"], ["f1", "f2"]))
    model.setSoftVoting(True)
    probs = model.predict_proba(pd.DataFrame({"f1": [np.nan], "f2": [np.nan]}))[0]
    for c, p in expected.items():
        assert probs[c] == pytest.approx(p)
    # Only the second split missing: the row goes left at f1 ('<'), then both leaves weighted 1/2.
    probs = model.predict_proba(pd.DataFrame({"f1": [0.0], "f2": [np.nan]}))[0]
    assert probs["B"] == pytest.approx(0.5) and probs["C"] == pytest.approx(0.5)


def test_tree_missing_key_and_nan_are_equivalent():
    tree = DecisionTreeMC(ASYM_TREE, ["A", "B", "C"])
    by_key = tree.useTree({"f2": 1.0})
    by_nan = tree.useTree(pd.Series({"f1": np.nan, "f2": 1.0}))
    assert by_key == pytest.approx(by_nan)
    assert by_key["A"] == pytest.approx(0.75) and by_key["B"] == pytest.approx(0.25)


def test_format1_tree_uses_equal_branch_weights():
    old = json.loads(json.dumps(ASYM_TREE).replace('"n_ge": 6, "n_lt": 2, ', ""))
    tree = DecisionTreeMC(old, ["A", "B", "C"])
    out = tree.useTree({})
    assert out["A"] == pytest.approx(0.5)


@pytest.mark.parametrize("engine", ENGINES)
def test_missing_value_probabilities_sum_to_one(engine):
    df = pd.read_csv("datasets/iris.csv")
    model = fit(make_model(engine, n_trees=8, target_col="variety", random_seed=1), df)
    model.setSoftVoting(True)
    rng = np.random.default_rng(0)
    X = df.drop(columns="variety").copy()
    X = X.mask(rng.random(X.shape) < 0.4)
    for probs in model.predict_proba(X):
        assert sum(probs.values()) == pytest.approx(1.0)


# --- Item 3: threshold floor ---


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("get_best_tree", [True, False])
def test_threshold_floor_on_noise(engine, get_best_tree, caplog):
    df = noise_dataset()
    model = make_model(
        engine, n_trees=8, target_col="target", get_best_tree=get_best_tree, max_tree_attempts=60, random_seed=5
    )
    with caplog.at_level(logging.WARNING):
        fit(model, df)
    tau_min = model._tau_min()
    assert len(model.data) == 8
    for tree in model.data:
        assert tree.selection_score >= tau_min or tree.hit_attempt_cap
        assert tree.attempts <= 60


@pytest.mark.parametrize("engine", ENGINES)
def test_attempt_cap_warns(engine, caplog):
    df = noise_dataset(seed=2, n=200)
    model = make_model(engine, n_trees=4, target_col="target", tau_floor_eps=0.4, max_tree_attempts=5, random_seed=1)
    with caplog.at_level(logging.WARNING):
        fit(model, df)
    assert all(tree.hit_attempt_cap for tree in model.data)
    assert any("max_tree_attempts" in rec.message for rec in caplog.records)


# --- Item 4/5: pools, scoring split, effective-depth sampling ---


@pytest.mark.parametrize("engine", ENGINES)
def test_validation_pool_and_scoring_split(engine):
    df = pd.read_csv("datasets/titanic.csv")[["Sex", "Age", "Pclass", "Fare", "Survived"]].dropna()
    model = make_model(
        engine, n_trees=8, target_col="Survived", val_pool_pclass=100, score_split=0.5, random_seed=0
    )
    fit(model, df)
    assert model._batch_sizes(min(df["Survived"].value_counts())) == (10, 50, 50)
    assert all(rec["s_score"] is not None for rec in model.candidate_log)
    assert all(tree.selection_score is not None for tree in model.data)


@pytest.mark.parametrize("engine", ENGINES)
def test_effective_depth_sampling(engine):
    df = synthetic_informative(n=300)
    model = make_model(engine, n_trees=8, target_col="target", feature_sampling="effective_depth", random_seed=0)
    fit(model, df)
    assert model.L_eff == 6
    for rec in model.candidate_log:
        assert len(rec["F"]) <= max(model.min_feature, model.L_eff)
    report = model.depthReport()
    assert {"k", "max_depth_reached", "used_is_prefix", "L_eff"} <= set(report.columns)


def test_n_train_scaling():
    model = PyRandomForestMC(target_col="target", n_train_frac=0.5, n_train_cap=40)
    assert model._batch_sizes(1000) == (40, 10, 0)
    assert model._batch_sizes(30) == (15, 10, 0)
    assert model._batch_sizes(12) == (10, 2, 0)


# --- Item 6: ensemble selection ---


def test_ensemble_select_prefers_complementary_candidates():
    y = np.array([0, 0, 1, 1])
    good = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    bad = good[:, ::-1]
    probs = np.stack([bad, good, bad, good])
    chosen = ensemble_select(probs, y, n_select=4, n_init=1)
    assert set(chosen) <= {1, 3}
    assert mean_pairwise_agreement(probs[[1, 3]]) == 1.0


@pytest.mark.parametrize("engine", ENGINES)
def test_ensemble_selection_fit(engine):
    df = pd.read_csv("datasets/titanic.csv")[["Sex", "Age", "Pclass", "Fare", "Survived"]].dropna()
    model = make_model(
        engine, n_trees=12, target_col="Survived", selection="ensemble", ensemble_pool_pclass=60,
        score_split=0.5, random_seed=0,
    )
    fit(model, df)
    assert 1 <= len(model.data) <= 12
    assert len(model.survived_scores) == len(model.data)
    assert 0.5 <= model.ensemble_score <= 1.0
    assert len(model.predict(df)) == len(df)


# --- Item 7: importance from all candidates ---


@pytest.mark.parametrize("engine", ENGINES)
def test_candidate_importance_ranks_informative_features(engine):
    df = synthetic_informative()
    model = make_model(engine, n_trees=48, target_col="target", val_pool_pclass=150, score_split=0.5, random_seed=2)
    fit(model, df)
    imp = model.featImportanceMC(n_boot=50)
    informative = imp.loc[["inf0", "inf1", "inf2"], "delta"].mean()
    noise = imp.loc[[f"noise{i}" for i in range(10)], "delta"].mean()
    assert informative > noise
    ridge = model.featPairImportanceMC(max_pair_features=6)
    assert set(ridge["main"]) == set(model.feature_cols)
    assert all(isinstance(pair, tuple) for pair in ridge["pairs"])


# --- Item 9: smoothing, priors, weights, threshold ---


@pytest.mark.parametrize("engine", ENGINES)
def test_leaf_alpha(engine):
    df = pd.read_csv("datasets/iris.csv")
    model = fit(make_model(engine, n_trees=6, target_col="variety", random_seed=4), df)
    model.setSoftVoting(True)
    base = model.predict_proba(df.drop(columns="variety"))
    model.leaf_alpha = 0.0
    if engine == "cpp":
        model._sync_params_to_cpp()
    assert model.predict_proba(df.drop(columns="variety")) == base
    model.leaf_alpha = 1.0
    if engine == "cpp":
        model._sync_params_to_cpp()
    smoothed = model.predict_proba(df.drop(columns="variety"))
    assert all(0 < p < 1 for dist in smoothed for p in dist.values())
    assert all(sum(dist.values()) == pytest.approx(1.0) for dist in smoothed)


def imbalanced_dataset(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.08).astype(int)
    X = rng.normal(size=(n, 4)) + y[:, None] * np.array([1.2, 0.8, 0.0, 0.0])
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df["target"] = y.astype(str)
    return df


def log_loss(probs, y):
    return -np.mean([np.log(max(dist[label], 1e-6)) for dist, label in zip(probs, y)])


@pytest.mark.parametrize("engine", ENGINES)
def test_prior_correction_improves_log_loss(engine):
    train, test = imbalanced_dataset(0), imbalanced_dataset(1)
    model = make_model(engine, n_trees=32, target_col="target", leaf_alpha=1.0, random_seed=0)
    fit(model, train)
    model.setSoftVoting(True)
    X, y = test.drop(columns="target"), test["target"].to_numpy()
    raw = log_loss(model.predict_proba(X), y)
    model.class_priors = "train"
    if engine == "cpp":
        model._sync_params_to_cpp()
    corrected = model.predict_proba(X)
    assert log_loss(corrected, y) < raw
    assert np.mean([dist["1"] for dist in corrected]) < 0.3


@pytest.mark.parametrize("engine", ENGINES)
def test_samme_weights_and_decision_threshold(engine):
    train, test = imbalanced_dataset(2), imbalanced_dataset(3)
    model = make_model(engine, n_trees=16, target_col="target", tree_weighting="samme", random_seed=0)
    fit(model, train)
    model.setWeightedTrees(True)
    model.setSoftVoting(True)
    weights = model._tree_weights()
    assert np.all(weights >= 0) and weights.sum() > 0
    thr = model.tune_decision_threshold(test)
    assert model.positive_class == "1" and 0.0 <= thr <= 1.0
    preds = model.predict(test.drop(columns="target"))
    assert set(preds) <= {"0", "1"}


# --- Serialization ---


@pytest.mark.parametrize("engine", ENGINES)
def test_load_v14_model(engine):
    with open("datasets/model_dict.json") as f:
        old = json.load(f)
    df = pd.read_csv("datasets/titanic.csv")[old["feature_cols"] + ["Survived"]].dropna()
    df["Pclass"] = df["Pclass"].astype(str)
    model = make_model(engine, target_col="Survived")
    model.dict2model(old)
    assert model.tree_format == 1
    preds = model.predict(df)
    assert len(preds) == len(df)
    acc = np.mean(np.array(preds) == df["Survived"].astype(str).to_numpy())
    assert acc > 0.6


def test_v14_model_same_predictions_in_both_engines():
    if not CPP_AVAILABLE:
        pytest.skip("C++ engine not available")
    with open("datasets/model_dict.json") as f:
        old = json.load(f)
    df = pd.read_csv("datasets/titanic.csv")[old["feature_cols"]].dropna()
    df["Pclass"] = df["Pclass"].astype(str)
    df.loc[df.index[:40], "Age"] = np.nan
    py, cpp = PyRandomForestMC(target_col="Survived"), CppRandomForestMC(target_col="Survived")
    py.dict2model(old)
    cpp.dict2model(old)
    for model in (py, cpp):
        model.setSoftVoting(True)
    for a, b in zip(py.predict_proba(df), cpp.predict_proba(df)):
        for c in a:
            assert a[c] == pytest.approx(b[c])


def test_mixed_formats_refuse_to_merge():
    with open("datasets/model_dict.json") as f:
        old = json.load(f)
    df = pd.read_csv("datasets/titanic.csv")[old["feature_cols"] + ["Survived"]].dropna()
    df["Pclass"] = df["Pclass"].astype(str)
    new = fit(PyRandomForestMC(n_trees=4, target_col="Survived"), df)
    legacy = PyRandomForestMC(target_col="Survived")
    legacy.dict2model(old)
    with pytest.raises(ValueError):
        new.mergeForest(legacy)
    new.mergeForest(legacy, allow_mixed_formats=True)
    assert len(new) == 4 + len(old["Forest"])


@pytest.mark.parametrize("engine", ENGINES)
def test_round_trip_keeps_v160_fields(engine):
    df = pd.read_csv("datasets/titanic.csv")[["Sex", "Age", "Pclass", "Embarked", "Survived"]].dropna()
    df["Pclass"] = df["Pclass"].astype(str)
    model = fit(
        make_model(engine, n_trees=6, target_col="Survived", categorical_split="ordered", leaf_alpha=0.5, random_seed=0),
        df,
    )
    d = json.loads(json.dumps(model.model2dict(include_candidates=True), default=str))
    other = make_model(engine, target_col="Survived")
    other.dict2model(d)
    assert other.leaf_alpha == 0.5 and other.categorical_split == "ordered" and other.tree_format == 2
    assert len(other.candidate_log) == len(model.candidate_log)
    assert other.predict(df) == model.predict(df)
