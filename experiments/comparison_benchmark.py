"""
Fair comparison benchmark: random-forest-mc (v1.6.0 C++ Engine) vs scikit-learn RandomForestClassifier.

Fairness Principles:
1. Out-of-Sample Generalization: 80/20 stratified train/test split. Both models are trained on the exact same
   training samples and evaluated on the exact same held-out test samples.
2. Fair Preprocessing: Encoders and imputers are fit strictly on the training set with zero test leakage.
3. Multi-Metric Evaluation: In addition to raw accuracy, measures Balanced Accuracy (unweighted mean recall),
   Macro F1, ROC-AUC, and the Generalization Gap (Train Acc - Test Acc) to account for class imbalance and overfitting.
4. Identical Hardware & Threading: Concurrency is strictly matched:
   - 8 cores for 8 trees
   - 16 cores for 16, 32, 64 trees
5. Equal Search Space: 12 candidate hyperparameter configurations evaluated per (dataset, n_trees) for both models.
6. Timing: High-resolution wall-clock profiling for model.fit() and model.predict().
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import OrdinalEncoder

# Suppress verbose root warnings during batch training
logging.getLogger().setLevel(logging.ERROR)

# Add src to path to import random_forest_mc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from random_forest_mc import RandomForestMC

DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))

DATASET_CONFIGS = {
    "iris": {
        "path": os.path.join(DATASETS_DIR, "iris.csv"),
        "target": "variety",
        "drop_cols": [],
        "multiclass": True,
    },
    "titanic": {
        "path": os.path.join(DATASETS_DIR, "titanic.csv"),
        "target": "Survived",
        "drop_cols": ["PassengerId", "Name", "Ticket", "Cabin"],
        "multiclass": False,
    },
    "secom": {
        "path": os.path.join(DATASETS_DIR, "secom.csv"),
        "target": "target",
        "drop_cols": [],
        "multiclass": False,
    },
    "creditcard": {
        "path": os.path.join(DATASETS_DIR, "creditcard.csv"),
        "target": "Class",
        "drop_cols": ["Time"],
        "multiclass": False,
    },
}

TREE_COUNTS = [8, 16, 32, 64]

RFMC_PARAMS = {
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "batch_train_pclass": [10, 20],
}

SKLEARN_PARAMS = {
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "criterion": ["gini", "entropy"],
}


def get_cores(n_trees: int) -> int:
    return 8 if n_trees == 8 else 16


def prepare_data(name: str):
    cfg = DATASET_CONFIGS[name]
    df = pd.read_csv(cfg["path"])

    if cfg["drop_cols"]:
        df = df.drop(columns=cfg["drop_cols"], errors="ignore")

    target_col = cfg["target"]

    if name == "titanic":
        df = df.dropna().copy()
        df["Pclass"] = df["Pclass"].astype(str)
    elif name == "secom":
        # Drop columns with > 50% missing
        thresh = len(df) * 0.5
        df = df.dropna(axis=1, thresh=thresh)

    # Stratified Train/Test split (80% train, 20% test)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[target_col]
    )
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Dataset-specific imputations (train-only fit)
    if name == "secom":
        medians = train_df.drop(columns=[target_col]).median()
        feat_cols = medians.index.tolist()
        train_df[feat_cols] = train_df[feat_cols].fillna(medians)
        test_df[feat_cols] = test_df[feat_cols].fillna(medians)

    # Prepare for RFMC: target as string
    train_rfmc = train_df.copy()
    test_rfmc = test_df.copy()
    train_rfmc[target_col] = train_rfmc[target_col].astype(str)
    test_rfmc[target_col] = test_rfmc[target_col].astype(str)

    # Prepare for Sklearn: robust numeric encoding of any non-numeric columns
    X_train_sk = train_df.drop(columns=[target_col]).copy()
    y_train_sk = train_df[target_col].copy()
    X_test_sk = test_df.drop(columns=[target_col]).copy()
    y_test_sk = test_df[target_col].copy()

    cat_cols = [c for c in X_train_sk.columns if not pd.api.types.is_numeric_dtype(X_train_sk[c])]
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_sk[cat_cols] = enc.fit_transform(X_train_sk[cat_cols]).astype(float)
        X_test_sk[cat_cols] = enc.transform(X_test_sk[cat_cols]).astype(float)

    return {
        "name": name,
        "target_col": target_col,
        "multiclass": cfg["multiclass"],
        "train_rfmc": train_rfmc,
        "test_rfmc": test_rfmc,
        "X_train_sk": X_train_sk,
        "y_train_sk": y_train_sk,
        "X_test_sk": X_test_sk,
        "y_test_sk": y_test_sk,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }


def compute_roc_auc_rfmc(y_true_str, probs_dict_list, class_vals, multiclass=False):
    try:
        classes = sorted([str(c) for c in class_vals])
        if multiclass:
            prob_mat = np.array(
                [[float(d.get(c, d.get(str(c), 0.0))) for c in classes] for d in probs_dict_list]
            )
            return float(roc_auc_score(y_true_str, prob_mat, multi_class="ovr"))
        else:
            pos_class = classes[1] if len(classes) > 1 else classes[0]
            pos_probs = np.array(
                [float(d.get(pos_class, d.get(str(pos_class), 0.0))) for d in probs_dict_list]
            )
            y_bin = (np.array(y_true_str) == pos_class).astype(int)
            return float(roc_auc_score(y_bin, pos_probs))
    except Exception as e:
        return float("nan")


def compute_roc_auc_sk(y_true, proba, multiclass=False, model_classes=None):
    try:
        if multiclass:
            return float(roc_auc_score(y_true, proba, multi_class="ovr"))
        else:
            if proba.ndim == 2 and proba.shape[1] == 2:
                pos_probs = proba[:, 1]
                classes = sorted(list(model_classes)) if model_classes is not None else sorted(np.unique(y_true))
                pos_class = classes[1]
                y_bin = (np.array(y_true) == pos_class).astype(int)
                return float(roc_auc_score(y_bin, pos_probs))
            return float(roc_auc_score(y_true, proba))
    except Exception as e:
        return float("nan")


def run_benchmark():
    all_trials = []
    best_rfmc_results = []
    best_sklearn_results = []
    summary_results = []

    total_start = time.perf_counter()

    for ds_name in DATASET_CONFIGS:
        print(f"\n{'='*75}\n[DATASET] Processing: {ds_name.upper()}\n{'='*75}")
        data = prepare_data(ds_name)
        target_col = data["target_col"]
        multiclass = data["multiclass"]

        y_test_raw = data["test_rfmc"][target_col].to_numpy()
        y_train_raw = data["train_rfmc"][target_col].to_numpy()
        y_test_str = [str(x) for x in y_test_raw]
        y_train_str = [str(x) for x in y_train_raw]

        print(f"  Samples: Train = {data['n_train']:,} | Test = {data['n_test']:,}")

        for n_trees in TREE_COUNTS:
            cores = get_cores(n_trees)
            print(f"\n--- Trees: {n_trees} (Threads/Cores: {cores}) ---")

            # =================================================================
            # 1. RANDOM FOREST MC (v1.6.0 C++ Engine)
            # =================================================================
            print("  [RFMC v1.6.0] Grid Search (12 configurations)...", end="", flush=True)
            best_rfmc_trial = None

            t_rfmc_start = time.perf_counter()
            for params in ParameterGrid(RFMC_PARAMS):
                model = RandomForestMC(
                    n_trees=n_trees,
                    target_col=target_col,
                    n_threads=cores,
                    random_seed=42,
                    engine="cpp",
                    **params,
                )

                t0 = time.perf_counter()
                model.fit(data["train_rfmc"])
                fit_time = time.perf_counter() - t0

                t0 = time.perf_counter()
                test_preds = model.predict(data["test_rfmc"])
                pred_time = time.perf_counter() - t0

                train_preds = model.predict(data["train_rfmc"])

                test_pred_str = [str(x) for x in test_preds]
                train_pred_str = [str(x) for x in train_preds]

                train_acc = accuracy_score(y_train_str, train_pred_str)
                test_acc = accuracy_score(y_test_str, test_pred_str)
                train_bal_acc = balanced_accuracy_score(y_train_str, train_pred_str)
                test_bal_acc = balanced_accuracy_score(y_test_str, test_pred_str)
                test_f1 = f1_score(y_test_str, test_pred_str, average="macro")

                # Probabilities & ROC-AUC
                test_probs_dict = model.predict_proba(data["test_rfmc"])
                test_auc = compute_roc_auc_rfmc(
                    y_test_str, test_probs_dict, model.class_vals, multiclass=multiclass
                )

                trial_info = {
                    "model": "random-forest-mc-v1.6.0",
                    "dataset": ds_name,
                    "n_trees": n_trees,
                    "cores": cores,
                    "params": str(params),
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "train_bal_acc": train_bal_acc,
                    "test_bal_acc": test_bal_acc,
                    "test_f1_macro": test_f1,
                    "test_roc_auc": test_auc,
                    "time_train": fit_time,
                    "time_predict": pred_time,
                    "throughput_pred": len(test_preds) / max(pred_time, 1e-9),
                    "overfit_gap_acc": train_acc - test_acc,
                    "overfit_gap_bal_acc": train_bal_acc - test_bal_acc,
                }
                all_trials.append(trial_info)

                if (
                    best_rfmc_trial is None
                    or test_bal_acc > best_rfmc_trial["test_bal_acc"]
                    or (
                        test_bal_acc == best_rfmc_trial["test_bal_acc"]
                        and test_acc > best_rfmc_trial["test_acc"]
                    )
                ):
                    best_rfmc_trial = trial_info

            t_rfmc_elapsed = time.perf_counter() - t_rfmc_start
            print(f" Done ({t_rfmc_elapsed:.2f}s total)")
            best_rfmc_results.append(best_rfmc_trial)
            print(
                f"    -> RFMC Best | BalAcc: {best_rfmc_trial['test_bal_acc']:.4f} | "
                f"Acc: {best_rfmc_trial['test_acc']:.4f} | F1: {best_rfmc_trial['test_f1_macro']:.4f} | "
                f"AUC: {best_rfmc_trial['test_roc_auc']:.4f} | Fit: {best_rfmc_trial['time_train']:.3f}s | "
                f"Pred: {best_rfmc_trial['time_predict']:.4f}s"
            )

            # =================================================================
            # 2. SCIKIT-LEARN RANDOM FOREST
            # =================================================================
            print("  [Scikit-Learn RF] Grid Search (12 configurations)...", end="", flush=True)
            best_sk_trial = None

            t_sk_start = time.perf_counter()
            for params in ParameterGrid(SKLEARN_PARAMS):
                model = RandomForestClassifier(
                    n_estimators=n_trees,
                    n_jobs=cores,
                    random_state=42,
                    **params,
                )

                t0 = time.perf_counter()
                model.fit(data["X_train_sk"], data["y_train_sk"])
                fit_time = time.perf_counter() - t0

                t0 = time.perf_counter()
                test_preds = model.predict(data["X_test_sk"])
                pred_time = time.perf_counter() - t0

                train_preds = model.predict(data["X_train_sk"])

                y_test_sk = data["y_test_sk"]
                y_train_sk = data["y_train_sk"]

                train_acc = accuracy_score(y_train_sk, train_preds)
                test_acc = accuracy_score(y_test_sk, test_preds)
                train_bal_acc = balanced_accuracy_score(y_train_sk, train_preds)
                test_bal_acc = balanced_accuracy_score(y_test_sk, test_preds)
                test_f1 = f1_score(y_test_sk, test_preds, average="macro")

                prob_mat = model.predict_proba(data["X_test_sk"])
                test_auc = compute_roc_auc_sk(
                    y_test_sk, prob_mat, multiclass=multiclass, model_classes=model.classes_
                )

                trial_info = {
                    "model": "scikit-learn",
                    "dataset": ds_name,
                    "n_trees": n_trees,
                    "cores": cores,
                    "params": str(params),
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "train_bal_acc": train_bal_acc,
                    "test_bal_acc": test_bal_acc,
                    "test_f1_macro": test_f1,
                    "test_roc_auc": test_auc,
                    "time_train": fit_time,
                    "time_predict": pred_time,
                    "throughput_pred": len(test_preds) / max(pred_time, 1e-9),
                    "overfit_gap_acc": train_acc - test_acc,
                    "overfit_gap_bal_acc": train_bal_acc - test_bal_acc,
                }
                all_trials.append(trial_info)

                if (
                    best_sk_trial is None
                    or test_bal_acc > best_sk_trial["test_bal_acc"]
                    or (
                        test_bal_acc == best_sk_trial["test_bal_acc"]
                        and test_acc > best_sk_trial["test_acc"]
                    )
                ):
                    best_sk_trial = trial_info

            t_sk_elapsed = time.perf_counter() - t_sk_start
            print(f" Done ({t_sk_elapsed:.2f}s total)")
            best_sklearn_results.append(best_sk_trial)
            print(
                f"    -> Sklearn Best | BalAcc: {best_sk_trial['test_bal_acc']:.4f} | "
                f"Acc: {best_sk_trial['test_acc']:.4f} | F1: {best_sk_trial['test_f1_macro']:.4f} | "
                f"AUC: {best_sk_trial['test_roc_auc']:.4f} | Fit: {best_sk_trial['time_train']:.3f}s | "
                f"Pred: {best_sk_trial['time_predict']:.4f}s"
            )

            # Combined Head-to-Head Entry
            summary_entry = {
                "dataset": ds_name,
                "n_trees": n_trees,
                "cores": cores,
                # Accuracy & Balance
                "rfmc_test_bal_acc": best_rfmc_trial["test_bal_acc"],
                "sklearn_test_bal_acc": best_sk_trial["test_bal_acc"],
                "rfmc_test_acc": best_rfmc_trial["test_acc"],
                "sklearn_test_acc": best_sk_trial["test_acc"],
                "rfmc_test_f1": best_rfmc_trial["test_f1_macro"],
                "sklearn_test_f1": best_sk_trial["test_f1_macro"],
                "rfmc_test_auc": best_rfmc_trial["test_roc_auc"],
                "sklearn_test_auc": best_sk_trial["test_roc_auc"],
                # Timing
                "rfmc_fit_time": best_rfmc_trial["time_train"],
                "sklearn_fit_time": best_sk_trial["time_train"],
                "train_speedup_rfmc": best_sk_trial["time_train"] / max(best_rfmc_trial["time_train"], 1e-6),
                "rfmc_pred_time": best_rfmc_trial["time_predict"],
                "sklearn_pred_time": best_sk_trial["time_predict"],
                "pred_speedup_rfmc": best_sk_trial["time_predict"] / max(best_rfmc_trial["time_predict"], 1e-6),
                # Overfitting Gap
                "rfmc_overfit_gap_acc": best_rfmc_trial["overfit_gap_acc"],
                "sklearn_overfit_gap_acc": best_sk_trial["overfit_gap_acc"],
                # Best parameters
                "rfmc_best_params": best_rfmc_trial["params"],
                "sklearn_best_params": best_sk_trial["params"],
            }
            summary_results.append(summary_entry)

    # Export CSVs
    df_rfmc = pd.DataFrame(best_rfmc_results)
    df_sklearn = pd.DataFrame(best_sklearn_results)
    df_summary = pd.DataFrame(summary_results)
    df_all = pd.DataFrame(all_trials)

    df_rfmc.to_csv(os.path.join(RESULTS_DIR, "results_rfmc.csv"), index=False)
    df_sklearn.to_csv(os.path.join(RESULTS_DIR, "results_sklearn.csv"), index=False)
    df_summary.to_csv(os.path.join(RESULTS_DIR, "results_summary.csv"), index=False)
    df_all.to_csv(os.path.join(RESULTS_DIR, "results_all_trials.csv"), index=False)

    total_time = time.perf_counter() - total_start
    print("\n" + "=" * 75)
    print(f"ALL BENCHMARKS COMPLETED SUCCESSFULLY IN {total_time:.2f}s!")
    print(f"Results saved to:")
    print(f"  - {os.path.join(RESULTS_DIR, 'results_rfmc.csv')}")
    print(f"  - {os.path.join(RESULTS_DIR, 'results_sklearn.csv')}")
    print(f"  - {os.path.join(RESULTS_DIR, 'results_summary.csv')}")
    print(f"  - {os.path.join(RESULTS_DIR, 'results_all_trials.csv')}")
    print("=" * 75)


if __name__ == "__main__":
    run_benchmark()
