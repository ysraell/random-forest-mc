import time
import pandas as pd
import numpy as np
from random_forest_mc.model import RandomForestMC

def generate_dataset(n_samples=1000, n_features=10):
    np.random.seed(42)
    data = np.random.randn(n_samples, n_features)
    columns = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns)
    df["target"] = np.random.randint(0, 2, n_samples).astype(str)
    return df

def benchmark():
    df = generate_dataset()
    model = RandomForestMC(n_trees=10, target_col="target")
    
    start_time = time.time()
    model.fit(df)
    end_time = time.time()
    
    print(f"Training time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()
