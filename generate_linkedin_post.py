import json

def generate_linkedin_post():
    with open("results_optimized.json", "r") as f:
        opt_results = json.load(f)
    
    with open("results_legacy.json", "r") as f:
        leg_results = json.load(f)
        
    opt_dict = {(r['dataset'], r['n_trees']): r['time'] for r in opt_results}
    leg_dict = {(r['dataset'], r['n_trees']): r['time'] for r in leg_results}
    
    print("# 🚀 Supercharging Random Forest with NumPy! ⚡\n")
    print("I recently refactored the `random-forest-mc` library to replace Pandas with NumPy for the core tree-building logic. The results are incredible!\n")
    print("Here is a benchmark comparing the new version against v1.3.0:\n")
    
    print("| Dataset | Trees | Legacy (s) | Optimized (s) | Speedup |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    # Sort keys for consistent output
    sorted_keys = sorted(opt_dict.keys())
    
    for key in sorted_keys:
        dataset, n_trees = key
        opt_time = opt_dict.get(key, -1)
        leg_time = leg_dict.get(key, -1)
        
        # Only include if both succeeded
        if opt_time > 0 and leg_time > 0:
            speedup = leg_time / opt_time
            print(f"| {dataset} | {n_trees} | {leg_time:.4f} | {opt_time:.4f} | **{speedup:.2f}x** |")

    print("\nCheck out the project on GitHub: https://github.com/ysraell/random-forest-mc\n")
    print("#Python #DataScience #MachineLearning #NumPy #Performance #OpenSource")

if __name__ == "__main__":
    generate_linkedin_post()
