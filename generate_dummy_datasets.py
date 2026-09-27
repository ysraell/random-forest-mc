import json
import pandas as pd
import numpy as np
import os

def generate_dummy_data():
    with open("tests/datasets_metadata.json", "r") as f:
        metadata = json.load(f)

    os.makedirs("datasets", exist_ok=True)

    for ds_name, params in metadata.items():
        print(f"Generating {ds_name}...")
        cols = params["ds_cols"]
        target = params["target_col"]
        path = "datasets" + params["csv_path"]
        
        n_samples = 2000 # Increased sample size for better benchmarking
        data = {}
        
        for col in cols:
            if "Age" in col:
                data[col] = np.random.randint(0, 100, n_samples)
            elif "Pclass" in col:
                data[col] = np.random.randint(1, 4, n_samples)
            elif "SibSp" in col:
                data[col] = np.random.randint(0, 9, n_samples)
            elif "Fare" in col:
                data[col] = np.random.uniform(0, 500, n_samples)
            elif "Amount" in col or col.startswith("V") or "width" in col or "length" in col:
                data[col] = np.random.randn(n_samples)
            elif "Sex" in col:
                data[col] = np.random.choice(["male", "female"], n_samples)
            elif "Embarked" in col:
                data[col] = np.random.choice(["S", "C", "Q"], n_samples)
            else:
                data[col] = np.random.randn(n_samples)
                
        if ds_name == "titanic":
            data[target] = np.random.randint(0, 2, n_samples)
        elif ds_name == "iris":
            data[target] = np.random.choice(["Setosa", "Versicolor", "Virginica"], n_samples)
        else:
            data[target] = np.random.randint(0, 2, n_samples)
            
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)
        print(f"Saved to {path}")

if __name__ == "__main__":
    generate_dummy_data()
