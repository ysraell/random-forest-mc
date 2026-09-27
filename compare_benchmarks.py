import json


def compare_benchmarks():
    with open("results_optimized.json", "r") as f:
        opt_results = json.load(f)
    
    with open("results_legacy.json", "r") as f:
        leg_results = json.load(f)
        
    # Convert to dict for easier lookup
    opt_dict = {(r['dataset'], r['n_trees']): r['time'] for r in opt_results}
    leg_dict = {(r['dataset'], r['n_trees']): r['time'] for r in leg_results}
    
    print("# Benchmark Comparison: v1.3.0 vs Optimized (NumPy)\n")
    print("| Dataset | Trees | Legacy (s) | Optimized (s) | Speedup |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    for key in opt_dict:
        dataset, n_trees = key
        opt_time = opt_dict.get(key, -1)
        leg_time = leg_dict.get(key, -1)
        
        if opt_time == -1:
            opt_str = "Failed"
        else:
            opt_str = f"{opt_time:.4f}"
            
        if leg_time == -1:
            leg_str = "Failed"
            speedup = "-"
        else:
            leg_str = f"{leg_time:.4f}"
            if opt_time > 0:
                speedup = f"{leg_time / opt_time:.2f}x"
            else:
                speedup = "-"
                
        print(f"| {dataset} | {n_trees} | {leg_str} | {opt_str} | {speedup} |")

if __name__ == "__main__":
    compare_benchmarks()
