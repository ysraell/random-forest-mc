# v1.6.0:

- **Correctness fixes (on by default; `legacy_v150=True` restores v1.5.0):**
  - Midpoint split thresholds (`split_rule`): the tie fallback and the two-sample case could train rows into one child and route them to the other at inference, leaving unreachable leaves.
  - Weighted missing-value traversal (`missing_strategy`): a missing feature now splits the path weight by the training fraction of each branch. The pure-Python engine previously returned only the first (`>=`) branch's leaf, and the two engines disagreed on NaN values and on unknown categories.
  - Threshold floor and attempt cap (`tau_floor_eps`, `max_tree_attempts`): Regime B can no longer lower the threshold below chance level, and Regime A no longer crashes when every candidate scores 0.
- **Method options (opt-in):** validation pool and separate scoring set (`val_pool_pclass`, `score_split`), effective-depth feature sampling and scaled training batches (`feature_sampling`, `n_train_frac`, `n_train_cap`), ensemble selection over all candidates (`selection="ensemble"`), label-aware and ordered-categorical splits (`split_supervision`, `split_lookahead`, `categorical_split`), leaf smoothing, prior correction and SAMME tree weights (`leaf_alpha`, `class_priors`, `tree_weighting`), and `tune_decision_threshold`.
- **Analysis:** every Monte Carlo candidate is logged (`candidate_log`); new `featImportanceMC`, `featPairImportanceMC` and `depthReport`.
- **Serialization:** tree format 2 (`n_ge`/`n_lt` split counts, leaf `n`, feature sequence, selection score); v1.5.0 and older dicts still load. Categorical splits are stored as category values in both engines, so C++ models round-trip categorical features and load older Python dicts. `dict2model(add=True)` now actually appends the trees. `mergeForest`/`drop_duplicated_trees` check `tree_format`.
- **Other:** C++ tree seeds no longer depend on thread scheduling; Python validation is vectorized; `featScoreMean` uses each tree's own score; `split_with_replace` is deprecated.
- **Ablation** (`experiments/ablation_v160.py`, C++ engine, 32 trees, soft voting, 10 seeds × 5-fold stratified CV; balanced accuracy, ±95% CI ≈ 0.005–0.01):

  | Change vs. v1.5.0 | iris | titanic | creditcard 1:20 | synthetic |
  | :-- | :-- | :-- | :-- | :-- |
  | v1.5.0 (`legacy_v150=True`) | 0.957 | 0.761 | 0.917 | 0.820 |
  | v1.6.0 defaults | 0.954 | 0.769 | 0.914 | 0.829 |
  | `n_train_frac=0.2, n_train_cap=200` | 0.954 | **0.785** | **0.923** | **0.858** |
  | `split_supervision=2` | 0.949 | 0.768 | 0.919 | 0.846 |
  | `val_pool_pclass=200, score_split=0.5` | 0.956 | 0.766 | 0.919 | 0.831 |
  | `selection="ensemble"` | 0.923 | 0.750 | 0.916 | 0.821 |

  - The defaults keep accuracy within the CI and lower log-loss (iris 0.131 → 0.108, creditcard 0.194 → 0.170, synthetic 0.464 → 0.454).
  - The validation pool with a scoring split gives the best calibration without priors (creditcard log-loss 0.131, ECE 0.151 → 0.094).
  - `class_priors="train"` gives the best creditcard log-loss (0.127) and ECE (0.030), but argmax decisions then collapse to the majority class (balanced accuracy 0.5): pair it with `tune_decision_threshold`.
  - `leaf_alpha=1` is too strong for leaves of 1–3 rows (log-loss rises); use smaller values.
  - Missing values (MCAR on test rows): `weighted` matches or beats `uniform_leaves` everywhere (iris at 50% masking: 0.823 vs 0.757), and both beat mean imputation on balanced accuracy.
  - `selection="ensemble"` is experimental: with `ensemble_metric="balanced_accuracy"` the greedy search stops early on the small holdout (≈ 14 distinct trees) and loses accuracy. `ensemble_metric="logloss"` keeps the full forest and nearly matches per-tree selection (titanic hold-out 0.798 vs 0.807).

# v1.5.0:

- **Modern C++ Backend (C++17 + nanobind):**
  - High-performance C++ core engine with $O(N)$ linear median splits via `std::nth_element`.
  - True GIL-free multi-threading (`std::thread`) for parallel tree planting and batch inference.
  - Substantial speedup: **~70x faster training** (`fit`) and **~304x faster inference** (`predict`).
- **Dual-Backend Support:**
  - Added `engine="auto" | "cpp" | "python"` parameter to `RandomForestMC` (defaults to `"auto"`).
  - Added dedicated C++ submodule: `random_forest_mc.cpp_model.RandomForestMC`.
  - Added direct exports: `RandomForestMCPy`, `RandomForestMCCPP`, and `RandomForestMCC`.
- **100% Serialization Parity:**
  - Full bidirectional compatibility between pure Python and C++ models: models saved via `model2dict()` can be loaded across engines via `dict2model()` with identical predictions.
- **Build & Packaging:**
  - Integrated `nanobind` compilation into Poetry with `build.py`.
  - Automatic graceful fallback to pure Python on systems without a C++ compiler.
  - Python 3.9 through 3.14 support.

# v1.4.0:

- ~~Refactor to use NumPy or built in Python features as core data operations.

# v1.3.0:

- Reorg the code in modules: tree, forest and tools. Move some methods from forest class to tools as a subclass.
- Change the `tqdm.contrib.concurrent` to `asyncio`: doesn't works well enough.
- Investigate how use the new ML-focused features from `Python 3.13` can be use here: bascally, with pandas working in the core os the planting tree, all optimization will not affect significantly. 
    - Experiment with JIT: no significnt differences.
    - Experiment with GIL off: no significnt differences.
- Add `Python 3.13` in the unit tests.

# v1.2.0:

- Fix bug with function `drop_duplicated_trees` that is keeping the duplicated only.
- Add in the `dict2model` the possibility to add and not only replace the current Forest. E.g.: dict2model(Dict, add = False) -> Default: False.
- Remove `split_train_val_replace`.
- Use the trees in parallel processing.
- Docstring.  
- Add parameter for active validation score (like loss) for each set of a given number of trees generated.

# v1.1.2

    1) Add the different Python versions in the automated tests.

# v1.1.1

    1) Add `NaN`, `None`, `Null` cheking before use and generate the Tree.
    2) Remove `self.dataset = dataset.dropna()` (l. 509, `model.py`).

# v1.1.0

    1) Fix utils.

# v1.1.0

    1) fix: `survived_score` getting wrong value. When the Tree is not dropped, the value got is `survived_score` and not th_val. Is not crictical, because generally the trees require many trees, so the score is got correctly.
    2) Add quality checks in `utils.LoadDicts`.
    3) Add input parameter `ignore_errors` in `utils.LoadDicts`.
    4) Add iterable protocol in `utils.LoadDicts`.
    5) Refactor some type hints.
    6) Add missing values prediction feature.

# v1.0.3

    1) Add the parameter `max_depth` to limit how deep the branching will be.
    2) Add the `min_samples_split`, once get this minimun or less, the leaf is generated instead a new branching.
    3) Fix the `__let__` that was with `>` instead `<`.
    4) Remove the threading mode for fit in paralell processing.

# v1.0.2

    1) Replace sum with fsum in many parts., from math standard lib. See: https://docs.python.org/3/library/math.html. 
