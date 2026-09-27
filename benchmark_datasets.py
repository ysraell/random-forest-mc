import json
import pandas as pd
import time
from random_forest_mc.model import RandomForestMC
import os

def benchmark_datasets(args):
    with open("tests/datasets_metadata.json", "r") as f:
        metadata = json.load(f)

    datasets_dir = "datasets"
    n_trees_list = [8, 16, 32, 64, 128]
    
    results = []

    for ds_name, params in metadata.items():
        print(f"Benchmarking {ds_name}...")
        csv_path = os.path.join(datasets_dir, params["csv_path"].lstrip("/"))
        target_col = params["target_col"]
        
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"File not found: {csv_path}")
            continue

        # Preprocess if needed (e.g. dropna as in tests)
        df = df.dropna().reset_index(drop=True)
        
        for n_trees in n_trees_list:
            max_workers = 8 if n_trees == 8 else 16
            
            print(f"  n_trees={n_trees}, max_workers={max_workers}")
            
            model = RandomForestMC(
                n_trees=n_trees, 
                target_col=target_col,
                max_discard_trees=n_trees * 4
            )
            
            start_time = time.time()
            try:
                model.fitParallel(df, max_workers=max_workers)
                end_time = time.time()
                duration = end_time - start_time
                print(f"    Time: {duration:.4f}s")
            except Exception as e:
                print(f"    Failed: {e}")
                duration = -1
            
            results.append({
                "dataset": ds_name,
                "n_trees": n_trees,
                "max_workers": max_workers,
                "time": duration
            })

    print("\nSummary Results:")
    print(f"{'Dataset':<25} {'Trees':<10} {'Workers':<10} {'Time (s)':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['dataset']:<25} {r['n_trees']:<10} {r['max_workers']:<10} {r['time']:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()
    benchmark_datasets(args)
