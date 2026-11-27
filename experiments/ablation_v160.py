"""
Ablation of the v1.6.0 changes against v1.5.0 behavior (`legacy_v150=True`).

One change at a time, repeated stratified K-fold over several seeds. Reports balanced accuracy,
macro-F1, log-loss, Brier score, expected calibration error, fit and predict time, plus a
missing-value study (MCAR masks on the test fold).

Usage:
    PYTHONPATH=src python experiments/ablation_v160.py --seeds 10 --folds 5 --engine cpp
    PYTHONPATH=src python experiments/ablation_v160.py --quick

scikit-learn and imbalanced-learn baselines run only when those packages are installed.
"""

import argparse
import json
import os
import time
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from random_forest_mc import RandomForestMC

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.abspath(os.path.join(HERE, "..", "datasets"))

# One change at a time on top of the v1.6.0 defaults; "v1.5.0" is the reference point.
CONFIGS: Dict[str, dict] = {
    "v1.5.0": {"legacy_v150": True},
    "v1.6.0 defaults": {},
    "split_rule=legacy": {"split_rule": "legacy"},
    "missing=uniform_leaves": {"missing_strategy": "uniform_leaves"},
    "no tau floor": {"tau_floor_eps": None, "max_tree_attempts": None},
    "val pool 200 + score split": {"val_pool_pclass": 200, "score_split": 0.5},
    "effective-depth sampling": {"feature_sampling": "effective_depth"},
    "n_train scaling": {"n_train_frac": 0.2, "n_train_cap": 200},
    "ensemble selection": {"selection": "ensemble", "ensemble_pool_pclass": 100, "score_split": 0.5},
    "split_supervision=1": {"split_supervision": 1},
    "split_supervision=2": {"split_supervision": 2},
    "categorical ordered": {"categorical_split": "ordered"},
    "leaf_alpha=1": {"leaf_alpha": 1.0},
    "leaf_alpha=1 + priors": {"leaf_alpha": 1.0, "class_priors": "train"},
    "samme weights": {"tree_weighting": "samme", "_weighted": True},
}


# --- Datasets ---


def load_datasets(quick: bool) -> Dict[str, pd.DataFrame]:
    out = {}
    iris = pd.read_csv(os.path.join(DATASETS_DIR, "iris.csv")).rename(columns={"variety": "target"})
    out["iris"] = iris
    titanic = pd.read_csv(os.path.join(DATASETS_DIR, "titanic.csv"))
    titanic = titanic[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]].dropna()
    titanic["Pclass"] = titanic["Pclass"].astype(str)
    out["titanic"] = titanic.rename(columns={"Survived": "target"})
    credit_path = os.path.join(DATASETS_DIR, "creditcard.csv")
    if os.path.exists(credit_path):
        credit = pd.read_csv(credit_path).drop(columns=["Time"])
        fraud = credit[credit["Class"] == 1]
        normal = credit[credit["Class"] == 0].sample(n=20 * len(fraud), random_state=0)  # 1:20 imbalance
        credit = pd.concat([fraud, normal]).sample(frac=1.0, random_state=0)
        if quick:
            credit = credit.sample(n=3000, random_state=0)
        out["creditcard_1to20"] = credit.rename(columns={"Class": "target"})
    out["synthetic_informative"] = synthetic_informative()
    return out


def synthetic_informative(n=2000, n_noise=10, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    inf = rng.normal(size=(n, 3))
    logit = 2.5 * inf[:, 0] - 2.0 * inf[:, 1] + 1.5 * inf[:, 2]
    y = (logit + rng.normal(scale=0.5, size=n) > 0).astype(int)
    df = pd.DataFrame(inf, columns=["inf0", "inf1", "inf2"])
    df["red0"] = inf[:, 0] + rng.normal(scale=0.3, size=n)
    df["red1"] = inf[:, 1] + rng.normal(scale=0.3, size=n)
    for i in range(n_noise):
        df[f"noise{i}"] = rng.normal(size=n)
    df["target"] = y
    return df


def stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    for c in np.unique(y):
        idx = rng.permutation(np.nonzero(y == c)[0])
        for i, chunk in enumerate(np.array_split(idx, n_folds)):
            folds[i].extend(chunk)
    return [np.array(f) for f in folds]


# --- Metrics ---


def metrics(y_true: np.ndarray, proba: np.ndarray, classes: List[str]) -> Dict[str, float]:
    y_idx = np.array([classes.index(c) for c in y_true])
    pred = np.argmax(proba, axis=1)
    recalls, f1s = [], []
    for k in range(len(classes)):
        tp = np.sum((pred == k) & (y_idx == k))
        fn = np.sum((pred != k) & (y_idx == k))
        fp = np.sum((pred == k) & (y_idx != k))
        if tp + fn:
            recalls.append(tp / (tp + fn))
        f1s.append(2 * tp / max(1, 2 * tp + fn + fp))
    p_true = np.clip(proba[np.arange(len(y_idx)), y_idx], 1e-6, 1.0)
    onehot = np.eye(len(classes))[y_idx]
    conf = proba.max(axis=1)
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = sum(
        np.sum(bins == b) / len(conf) * abs(np.mean(pred[bins == b] == y_idx[bins == b]) - np.mean(conf[bins == b]))
        for b in range(10)
        if np.any(bins == b)
    )
    return {
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "log_loss": float(-np.mean(np.log(p_true))),
        "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
        "ece": float(ece),
    }


def proba_matrix(dists: List[dict], classes: List[str]) -> np.ndarray:
    return np.array([[float(d.get(c, 0.0)) for c in classes] for d in dists])


# --- Models ---


def rfmc_runner(engine: str, n_trees: int, params: dict) -> Callable:
    params = dict(params)
    weighted = params.pop("_weighted", False)

    def run(train: pd.DataFrame, test_X: pd.DataFrame, classes: List[str], seed: int):
        kwargs = dict(n_trees=n_trees, target_col="target", engine=engine, **params)
        if engine != "python":
            kwargs["random_seed"] = seed
        else:
            np.random.seed(seed)
        model = RandomForestMC(**kwargs)
        t0 = time.perf_counter()
        model.fit(train.copy())
        fit_time = time.perf_counter() - t0
        model.setSoftVoting(True)
        model.setWeightedTrees(weighted)
        t0 = time.perf_counter()
        proba = proba_matrix(model.predict_proba(test_X), classes)
        return proba, fit_time, time.perf_counter() - t0, model

    return run


def sklearn_runners(n_trees: int) -> Dict[str, Callable]:
    runners = {}
    try:
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    except ImportError:
        return runners
    try:
        from imblearn.ensemble import BalancedRandomForestClassifier
    except ImportError:
        BalancedRandomForestClassifier = None

    def make(factory):
        def run(train, test_X, classes, seed):
            X = pd.get_dummies(train.drop(columns="target"), dtype=float)
            Xt = pd.get_dummies(test_X, dtype=float).reindex(columns=X.columns, fill_value=0.0)
            model = factory(seed)
            t0 = time.perf_counter()
            model.fit(X, train["target"].astype(str))
            fit_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            proba = model.predict_proba(Xt.fillna(X.mean()))
            order = [list(model.classes_).index(c) for c in classes]
            return proba[:, order], fit_time, time.perf_counter() - t0, model

        return run

    runners["sklearn RF"] = make(lambda s: RandomForestClassifier(n_estimators=n_trees, random_state=s, n_jobs=-1))
    runners["sklearn RF balanced"] = make(
        lambda s: RandomForestClassifier(n_estimators=n_trees, class_weight="balanced", random_state=s, n_jobs=-1)
    )
    runners["sklearn ExtraTrees"] = make(lambda s: ExtraTreesClassifier(n_estimators=n_trees, random_state=s, n_jobs=-1))
    runners["sklearn ExtraTrees balanced"] = make(
        lambda s: ExtraTreesClassifier(n_estimators=n_trees, class_weight="balanced", random_state=s, n_jobs=-1)
    )
    if BalancedRandomForestClassifier is not None:
        runners["imblearn BalancedRF"] = make(
            lambda s: BalancedRandomForestClassifier(
                n_estimators=n_trees, random_state=s, n_jobs=-1, sampling_strategy="all", replacement=True, bootstrap=False
            )
        )
    return runners


# --- Experiments ---


def run_ablation(datasets, runners, seeds: int, folds: int) -> List[dict]:
    rows = []
    for ds_name, df in datasets.items():
        df = df.copy()
        df["target"] = df["target"].astype(str)
        classes = sorted(df["target"].unique())
        y = df["target"].to_numpy()
        for seed in range(seeds):
            for k, test_idx in enumerate(stratified_folds(y, folds, seed)):
                train = df.drop(index=df.index[test_idx])
                test = df.iloc[test_idx]
                test_X = test.drop(columns="target")
                for name, run in runners.items():
                    proba, fit_t, pred_t, _ = run(train, test_X, classes, seed * 100 + k)
                    row = {"dataset": ds_name, "model": name, "seed": seed, "fold": k}
                    row.update(metrics(test["target"].to_numpy(), proba, classes))
                    row.update({"fit_time": fit_t, "predict_time": pred_t})
                    rows.append(row)
            print(f"[ablation] {ds_name}: seed {seed + 1}/{seeds} done", flush=True)
    return rows


def run_missing(datasets, engine: str, n_trees: int, seeds: int, folds: int) -> List[dict]:
    rows = []
    strategies = {
        "weighted": {"missing_strategy": "weighted"},
        "uniform_leaves": {"missing_strategy": "uniform_leaves"},
        "mean imputation": {"missing_strategy": "weighted", "_impute": True},
    }
    for ds_name, df in datasets.items():
        df = df.copy()
        df["target"] = df["target"].astype(str)
        classes = sorted(df["target"].unique())
        y = df["target"].to_numpy()
        for seed in range(seeds):
            test_idx = stratified_folds(y, folds, seed)[0]
            train = df.drop(index=df.index[test_idx])
            test = df.iloc[test_idx]
            run = rfmc_runner(engine, n_trees, {})
            _, _, _, model = run(train, test.drop(columns="target"), classes, seed)
            rng = np.random.default_rng(seed)
            for rate in (0.1, 0.3, 0.5):
                X = test.drop(columns="target")
                masked = X.mask(rng.random(X.shape) < rate)
                for name, opts in strategies.items():
                    model.missing_strategy = opts["missing_strategy"]
                    if hasattr(model, "_sync_params_to_cpp"):
                        model._sync_params_to_cpp()
                    X_eval = masked
                    if opts.get("_impute"):
                        X_eval = masked.copy()
                        for col in X_eval.columns:
                            if col in model.numeric_cols:
                                X_eval[col] = X_eval[col].fillna(train[col].mean())
                            else:
                                X_eval[col] = X_eval[col].fillna(train[col].mode().iloc[0])
                    proba = proba_matrix(model.predict_proba(X_eval), classes)
                    row = {"dataset": ds_name, "strategy": name, "mask_rate": rate, "seed": seed}
                    row.update(metrics(test["target"].to_numpy(), proba, classes))
                    rows.append(row)
    return rows


def to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown()
    except ImportError:  # 'tabulate' not installed
        return "```\n" + df.to_string() + "\n```"


def summarize(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    value_cols = [c for c in df.columns if c not in by + ["seed", "fold"]]
    grouped = df.groupby(by)[value_cols]
    mean, sem, n = grouped.mean(), grouped.sem(), grouped.count()
    out = mean.copy()
    for col in value_cols:
        half = 1.96 * sem[col].fillna(0.0)
        out[col] = [f"{m:.4f} ± {h:.4f}" for m, h in zip(mean[col], half)]
    out["n"] = n[value_cols[0]]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-trees", type=int, default=32)
    parser.add_argument("--engine", default="auto", choices=["auto", "cpp", "python"])
    parser.add_argument("--quick", action="store_true", help="2 seeds, 3 folds, subsampled data")
    parser.add_argument("--out", default=os.path.join(HERE, "ablation_v160_results"))
    args = parser.parse_args()
    if args.quick:
        args.seeds, args.folds = 2, 3

    datasets = load_datasets(args.quick)
    runners = {name: rfmc_runner(args.engine, args.n_trees, params) for name, params in CONFIGS.items()}
    runners.update(sklearn_runners(args.n_trees))

    os.makedirs(args.out, exist_ok=True)
    ablation = pd.DataFrame(run_ablation(datasets, runners, args.seeds, args.folds))
    missing = pd.DataFrame(run_missing(datasets, args.engine, args.n_trees, args.seeds, args.folds))
    ablation.to_csv(os.path.join(args.out, "ablation_raw.csv"), index=False)
    missing.to_csv(os.path.join(args.out, "missing_raw.csv"), index=False)

    summary = summarize(ablation, ["dataset", "model"])
    summary_missing = summarize(missing, ["dataset", "strategy", "mask_rate"])
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("# RF-TSMC v1.6.0 ablation\n\n")
        f.write(f"seeds={args.seeds}, folds={args.folds}, n_trees={args.n_trees}, engine={args.engine}\n\n")
        f.write("Mean ± 95% CI (normal approximation over seeds × folds).\n\n## Ablation\n\n")
        f.write(to_markdown(summary))
        f.write("\n\n## Missing values (MCAR on the test fold)\n\n")
        f.write(to_markdown(summary_missing))
        f.write("\n")
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"args": vars(args), "configs": CONFIGS}, f, indent=2)
    print(summary.to_string())
    print(summary_missing.to_string())
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
