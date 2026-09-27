import pandas as pd
import numpy as np
import time
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Add src to path to import random_forest_mc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from random_forest_mc.model import RandomForestMC

# Configuration
DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

DATASETS = {
    "secom": {
        "path": os.path.join(DATASETS_DIR, "secom.csv"),
        "target": "target",
        "drop_cols": []
    }
}

TREE_COUNTS = [8, 16, 32, 64]

RFMC_PARAMS = {
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "batch_train_pclass": [10, 20],
    "max_discard_trees": [32, 64]
}

SKLEARN_PARAMS = {
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "criterion": ["gini", "entropy"]
}

def load_dataset(name):
    config = DATASETS[name]
    df = pd.read_csv(config["path"])
    
    # Basic preprocessing
    if name == "secom":
        # Drop columns with too many missing values (>50%)
        threshold = len(df) * 0.5
        df = df.dropna(axis=1, thresh=threshold)
        # Fill remaining NaN with median
        df = df.fillna(df.median())

    if config["drop_cols"]:
        df = df.drop(columns=config["drop_cols"], errors='ignore')

    return df, config["target"]

def get_cores(n_trees):
    if n_trees == 8:
        return 8
    return 16 # "16+ for 16+" implies at least 16, user said "16+" so 16 is safe/good.

def run_benchmark():
    results_rfmc = []
    results_sklearn = []

    for ds_name in DATASETS:
        print(f"Processing dataset: {ds_name}")
        df, target_col = load_dataset(ds_name)
        
        # Prepare data for Sklearn (needs numeric)
        df_sklearn = df.copy()
        le_dict = {}
        for col in df_sklearn.columns:
            if df_sklearn[col].dtype == 'object':
                le = LabelEncoder()
                df_sklearn[col] = le.fit_transform(df_sklearn[col].astype(str))
                le_dict[col] = le

        X_sklearn = df_sklearn.drop(columns=[target_col])
        y_sklearn = df_sklearn[target_col]

        # Prepare data for RFMC
        # RFMC expects target to be string usually for classification in the README example
        # but it can handle it. Let's ensure target is string for RFMC as per README.
        df_rfmc = df.copy()
        df_rfmc[target_col] = df_rfmc[target_col].astype(str)

        for n_trees in TREE_COUNTS:
            cores = get_cores(n_trees)
            print(f"  Trees: {n_trees}, Cores: {cores}")

            # --- RandomForestMC ---
            print("    Running RandomForestMC Grid Search...")
            best_acc_rfmc = -1
            best_res_rfmc = None
            
            for params in ParameterGrid(RFMC_PARAMS):
                # Train
                model = RandomForestMC(
                    n_trees=n_trees, 
                    target_col=target_col, 
                    **params
                )
                
                start_time = time.time()
                try:
                    model.fitParallel(dataset=df_rfmc, max_workers=cores, disable_progress_bar=True)
                    train_time = time.time() - start_time
                    
                    # Predict
                    start_time = time.time()
                    # Using hard voting by default
                    y_pred = model.testForest(df_rfmc) 
                    predict_time = time.time() - start_time
                    
                    # Accuracy
                    y_true = df_rfmc[target_col].to_list()
                    acc = accuracy_score(y_true, y_pred)
                    
                    if acc > best_acc_rfmc:
                        best_acc_rfmc = acc
                        best_res_rfmc = {
                            "dataset": ds_name,
                            "n_trees": n_trees,
                            "params": str(params),
                            "accuracy": acc,
                            "time_train": train_time,
                            "time_predict": predict_time,
                            "cores": cores
                        }
                except Exception as e:
                    print(f"      RFMC Error with params {params}: {e}")

            if best_res_rfmc:
                results_rfmc.append(best_res_rfmc)
                print(f"      Best RFMC Acc: {best_res_rfmc['accuracy']:.4f}")

            # --- Sklearn RandomForest ---
            print("    Running Sklearn RandomForest Grid Search...")
            best_acc_sklearn = -1
            best_res_sklearn = None

            for params in ParameterGrid(SKLEARN_PARAMS):
                model = RandomForestClassifier(
                    n_estimators=n_trees,
                    n_jobs=cores,
                    **params
                )
                
                start_time = time.time()
                model.fit(X_sklearn, y_sklearn)
                train_time = time.time() - start_time
                
                start_time = time.time()
                y_pred = model.predict(X_sklearn)
                predict_time = time.time() - start_time
                
                acc = accuracy_score(y_sklearn, y_pred)
                
                if acc > best_acc_sklearn:
                    best_acc_sklearn = acc
                    best_res_sklearn = {
                        "dataset": ds_name,
                        "n_trees": n_trees,
                        "params": str(params),
                        "accuracy": acc,
                        "time_train": train_time,
                        "time_predict": predict_time,
                        "cores": cores
                    }
            
            if best_res_sklearn:
                results_sklearn.append(best_res_sklearn)
                print(f"      Best Sklearn Acc: {best_res_sklearn['accuracy']:.4f}")

    # Save results
    pd.DataFrame(results_rfmc).to_csv(os.path.join(RESULTS_DIR, "results_rfmc.csv"), index=False)
    pd.DataFrame(results_sklearn).to_csv(os.path.join(RESULTS_DIR, "results_sklearn.csv"), index=False)
    print("Results saved.")

if __name__ == "__main__":
    run_benchmark()
