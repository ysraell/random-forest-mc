# Random Forest with Dynamic Tree Selection Based on Monte Carlo Simulation (RF-TSMC): Theoretical Formulation and Algorithmic Methodology

**Method Specification for [`random-forest-mc`](https://github.com/ysraell/random-forest-mc) (v1.6.0)**  
*Formalization of the Architecture, Learning Dynamics, Inference Rules, and Structural Explainability*

---

## Abstract

Standard Random Forest algorithms rely on bootstrap aggregating (bagging) paired with greedy, node-wise impurity minimization over the full training distribution. While effective in balanced, well-populated settings, this paradigm exhibits systematic sensitivity to class imbalance, incurs substantial split-search complexity across candidate thresholds, and optimizes decision trees strictly through local, myopic criteria. We formalize **Random Forest with Tree Selection Monte Carlo Based (RF-TSMC)**, a multiclass ensemble learning methodology that decouples local tree construction from global structural optimization. During induction, each candidate tree is grown from a small, strictly **class-balanced micro-batch** using an **impurity-free cyclic feature-queue heuristic** that applies deterministic count-median midpoint splits between distinct values (for numeric attributes) and modal one-vs.-rest splits (for categorical attributes), with optional label-aware variants. Structural optimization is shifted from greedy per-node search to an outer **Monte Carlo acceptance–rejection loop** over the joint space of random feature-subspace cardinalities and feature permutations, scored against an independent, class-balanced out-of-batch selection set under either a bounded-patience best-of-$`R_{\max}`$ regime or an adaptive decaying-threshold regime, both bounded below by an above-chance threshold floor; optionally, the forest is instead assembled by greedy ensemble selection over all Monte Carlo candidates. At inference time, missing or omitted feature values are handled natively without prior imputation via **dual-branch tree replication**, weighting the reachable leaves by the training fractions of the branches taken (C4.5-style marginalization) before ensemble combination through hard or soft voting, optionally weighted by each tree's survival score and corrected for class priors. Finally, because survival in the ensemble directly reflects discriminative utility across randomized subspaces, the resulting forest induces built-in marginal, pairwise, and instance-conditional feature importance measures, as well as counterfactual missing-value probability profiling and compositional forest merging.

---

## Changes in v1.6.0

Version 1.6.0 corrects several errors in the v1.5.0 specification and implementation identified by an external review, and adds opt-in extensions. The Python and C++ engines implement every rule below identically.

**Changed defaults (bug fixes).**
- `split_rule="midpoint"` (§5.1): numeric thresholds are midpoints between adjacent *distinct* values, so the inference rule $`x_f \ge \theta`$ reproduces the training partition exactly and every leaf is reachable. The v1.5.0 rule is `split_rule="legacy"`.
- `missing_strategy="weighted"` (§7.1): missing features are marginalized with branch weights equal to the training fractions $`n_{\ge}/n`$ and $`n_{<}/n`$. The v1.5.0 rule is `missing_strategy="uniform_leaves"`.
- `tau_floor_eps=0.05` (§6.2): acceptance requires a selection score of at least $`\tau_{\min} = 1/K + \varepsilon`$; batches that cannot reach it are redrawn. The v1.5.0 rule is `tau_floor_eps=None`.
- `max_tree_attempts="auto"` (§6.2): at most $`10\,R_{\max}`$ candidates per tree slot when the floor is active (unlimited in legacy mode).

**Opt-in extensions (defaults reproduce v1.5.0 behavior).** `split_supervision`, `split_lookahead`, `categorical_split` (§5.5); `feature_sampling` (§4); `n_train_frac`, `n_train_cap`, `val_pool_pclass`, `score_split` (§3, §6.1); `leaf_alpha`, `class_priors`, `tree_weighting`, `tune_decision_threshold` (§7.2); `selection`, `ensemble_pool_pclass`, `ensemble_init`, `ensemble_metric` (§6.4); candidate-based feature importance (§8.1); and the `tree_format` serialization tag (§8.3).

Passing `legacy_v150=True` restores every v1.5.0 behavior changed by the bug fixes (split rule, missing-value traversal, threshold floor and attempt cap).

---

## 1. Introduction and Motivation

Decision tree ensembles have long served as foundational non-parametric models for tabular classification. In classical **Random Forests (RF)** (Breiman, 2001), generalization arises from combining low-bias, high-variance trees trained on bootstrap replicates of the dataset $`\mathcal{D}`$, with feature subsampling ($`m_{\text{try}}`$) performed independently at each internal split node. In **Extremely Randomized Trees (Extra-Trees)** (Geurts et al., 2006), variance reduction is pushed further by drawing split thresholds at random rather than optimizing them exhaustively, while still evaluating an impurity score across candidate features at every node.

Despite their empirical success, classical CART-based forests share three structural constraints:

1. **Susceptibility to Class Imbalance:** Uniform bootstrap sampling replicates the empirical class distribution of $`\mathcal{D}`$. Under severe skew (e.g., rare-event or fraud detection), majority-class instances dominate both node-level impurity criteria (Gini index or Shannon entropy) and leaf posterior frequencies, degrading minority-class recall unless external resampling or cost weighting is introduced.
2. **Greedy Local Split Search:** Evaluating impurity reductions across candidate attributes and thresholds couples tree-construction cost to node sample size and feature dimensionality, while enforcing a myopic, top-down greedy heuristic that cannot evaluate the holistic discriminative power of an entire feature ordering before committing to early splits.
3. **Lack of Tree-Level Validation Filtering:** In standard bagging, every induced tree enters the ensemble regardless of out-of-sample performance on unseen samples; poorly generalizing trees created by unfavorable feature draws dilute ensemble votes with uniform weight.

**RF-TSMC** addresses these constraints through an inverted architectural principle:

> *Replace greedy, impurity-driven node optimization on skewed full-data bootstraps with ultra-fast, impurity-free deterministic splitting on class-balanced micro-batches, and delegate structural feature-subset and feature-ordering optimization to a Monte Carlo validation-selection process.*

Specifically, each tree in RF-TSMC is deliberately specialized (grown deeply on a small, class-balanced training subset $`\mathcal{I}_T`$ along a randomly ordered feature sequence $`F`$), then accepted or rejected via Monte Carlo trials against a disjoint, class-balanced validation subset $`\mathcal{I}_V`$. Only trees whose feature subspace and split sequence generalize to $`\mathcal{I}_V`$ survive into the forest.

---

## 2. Preliminaries and Notation

Let $`\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{M}`$ denote a supervised tabular dataset of $`M`$ complete observations (incomplete rows containing missing entries are excluded prior to training batch construction). Each observation $`\mathbf{x}_i \in \mathcal{X}`$ is described by $`d`$ predictor attributes $`\mathcal{F} = \{f_1, \dots, f_d\}`$ partitioned into:
- numeric features $`\mathcal{F}_{\text{num}} \subseteq \mathcal{F}`$, and
- categorical features $`\mathcal{F}_{\text{cat}} = \mathcal{F} \setminus \mathcal{F}_{\text{num}}`$.

The target variable takes values in a discrete label set $`\mathcal{C} = \{c_1, \dots, c_K\}`$ with $`K = |\mathcal{C}| \ge 2`$. For each class $`c \in \mathcal{C}`$, let $`\mathcal{I}^{(c)} = \{i \in \{1, \dots, M\} : y_i = c\}`$ denote its index set and $`M_c = |\mathcal{I}^{(c)}|`$ its cardinality, with minority count $`M_{\min} = \min_{c \in \mathcal{C}} M_c`$.

Table 1 summarizes the model hyperparameters and notation used throughout this formulation.

| Symbol | Parameter Name | Default | Theoretical Role |
| :--- | :--- | :---: | :--- |
| $`B`$ | `n_trees` | $`16`$ | Target number of surviving trees in the ensemble $`\mathcal{E}`$ |
| $`n_{\text{train}}`$ | `batch_train_pclass` | $`10`$ | Target number of training samples drawn per class per tree |
| $`n_{\text{val}}`$ | `batch_val_pclass` | $`10`$ | Target number of validation samples drawn per class per tree |
| $`f_{\min}, f_{\max}`$ | `min_feature`, `max_feature` | $`2, d`$ | Lower and upper bounds on sampled feature-subset size $`k`$ |
| $`\tau_0`$ | `th_start` | $`1.0`$ | Initial validation accuracy acceptance threshold |
| $`\Delta\tau`$ | `delta_th` | $`0.1`$ | Step decrement applied when relaxing $`\tau`$ (`get_best_tree=False`) |
| $`R_{\max}`$ | `max_discard_trees` | $`10`$ | Rejection patience budget before best-fallback or threshold decay |
| $`\beta_{\text{best}}`$ | `get_best_tree` | $`\text{True}`$ | Selection regime: bounded best-of-$`R_{\max}`$ vs. adaptive decay |
| $`d_{\max}`$ | `max_depth` | $`\infty^{\dagger}`$ | Maximum tree depth (root defined at depth $`t = 1`$) |
| $`s_{\min}`$ | `min_samples_split` | $`1`$ | Minimum sample cardinality required in *both* child partitions |
| $`\mathbb{I}_{\text{temp}}`$ | `temporal_features` | $`\text{False}`$ | Enforces ascending ordinal/temporal ordering on sampled $`F`$ |
| $`\sigma_{\text{soft}}, \sigma_{\text{wt}}`$ | `soft_voting`, `weighted_tree` | $`\text{False}, \text{False}`$ | Inference aggregation switches (soft vs. hard; score-weighted vs. uniform) |
| — | `split_rule` | `"midpoint"` | Numeric threshold rule: distinct-value midpoint (§5.1) or v1.5.0 `"legacy"` |
| $`\lambda_{\text{sup}}`$ | `split_supervision` | $`0`$ | Label-aware split level $`\in \{0, 1, 2\}`$ (§5.5) |
| $`H`$ | `split_lookahead` | $`2`$ | Queue features evaluated per node when $`\lambda_{\text{sup}} = 2`$ |
| — | `categorical_split` | `"mode"` | Categorical rule: mode-vs.-rest or `"ordered"` set split (§5.5) |
| — | `feature_sampling` | `"uniform"` | Cardinality law for $`k`$: `"uniform"` or `"effective_depth"` (§4) |
| $`\rho, n_{\text{cap}}`$ | `n_train_frac`, `n_train_cap` | None, None | Optional growth of $`n_{\text{train}}`$ with $`M_{\min}`$ (§3) |
| $`n_{\text{pool}}`$ | `val_pool_pclass` | $`n_{\text{val}}`$ | Per-class size of the out-of-batch validation pool (§3) |
| $`\gamma`$ | `score_split` | $`0`$ | Fraction of the pool reserved for the scoring set (§3, §6.1) |
| $`\varepsilon`$ | `tau_floor_eps` | $`0.05`$ | Threshold floor $`\tau_{\min} = 1/K + \varepsilon`$; None = legacy (§6.2) |
| $`A_{\max}`$ | `max_tree_attempts` | `"auto"` | Candidate cap per slot: $`10\,R_{\max}`$ with floor, unlimited in legacy (§6.2) |
| — | `missing_strategy` | `"weighted"` | Missing-value traversal: `"weighted"` or `"uniform_leaves"` (§7.1) |
| $`\alpha`$ | `leaf_alpha` | $`0`$ | Laplace smoothing of leaf distributions (§5.2, §7.2) |
| $`\pi`$ | `class_priors` | None | Prior correction of the forest output: None, `"train"`, or dict (§7.2) |
| — | `tree_weighting` | `"score"` | Tree weights when $`\sigma_{\text{wt}}`$: raw score or `"samme"` (§7.2) |
| — | `selection` | `"per_tree"` | Forest assembly: per-slot acceptance or greedy `"ensemble"` selection (§6.4) |
| $`n_{\text{ens}}, B_0`$ | `ensemble_pool_pclass`, `ensemble_init` | $`100, 3`$ | Ensemble holdout size per class and initial ensemble size (§6.4) |
| — | `ensemble_metric` | `"balanced_accuracy"` | Ensemble-selection objective: balanced accuracy or `"logloss"` (§6.4) |
| — | `legacy_v150` | $`\text{False}`$ | Restores all v1.5.0 defaults changed in v1.6.0 |

*† Unbounded up to implementation recursion/depth limits ($`1000`$ in the C++ backend; system recursion limit in pure Python).*

---

## 3. Class-Balanced Micro-Batch Partitioning

For each tree-selection episode $`b \in \{1, \dots, B\}`$, RF-TSMC constructs a fresh pair of disjoint, class-balanced training and validation index sets $`(\mathcal{I}_T, \mathcal{I}_V)`$ without replacement. In v1.6.0 the validation set $`\mathcal{I}_V`$ is a balanced out-of-batch *pool* that is further split into a selection set and a scoring set.

**Training batch.** The per-class training size is $`n_{\text{train}}`$, optionally grown with the minority class count:

$$
\tilde{n}_ {\text{train}} = \max\!\Big(n_{\text{train}},\; \min\!\big(\lfloor \rho\, M_{\min} \rfloor,\; n_{\text{cap}},\; M_{\min} - n_{\text{val}}\big)\Big), \qquad n'_ {\text{train}} = \max\!\big(1, \min(\tilde{n}_ {\text{train}}, M_{\min})\big),
$$

where $`\rho`$ = `n_train_frac` and $`n_{\text{cap}}`$ = `n_train_cap` (an absent cap is $`+\infty`$; without $`\rho`$, $`\tilde{n}_{\text{train}} = n_{\text{train}}`$, as in v1.5.0). The growth option exists because consistency results for median forests require the subsample size to grow with $`M`$ (Duroux & Scornet, 2018); a fixed $`n_{\text{train}} = 10`$ keeps every tree a function of $`10K`$ rows regardless of $`M`$.

**Validation pool.** The per-class pool size is

$$
n'_ {\text{pool}} = \max\!\big(0,\; \min(n_{\text{pool}},\; M_{\min} - n'_ {\text{train}})\big),
$$

where $`n_{\text{pool}}`$ = `val_pool_pclass` (default $`n_{\text{val}}`$). The pool is divided by the fraction $`\gamma`$ = `score_split` into $`n'_{\text{sc}} = \lfloor \gamma\, n'_{\text{pool}} \rfloor`$ scoring rows (at most $`n'_{\text{pool}} - 1`$) and $`n'_{\text{sel}} = n'_{\text{pool}} - n'_{\text{sc}}`$ selection rows per class; we write $`n'_{\text{val}} = n'_{\text{pool}}`$ for the total per-class validation size. With the defaults ($`n_{\text{pool}} = n_{\text{val}}`$, $`\gamma = 0`$), $`n'_{\text{sel}} = n'_{\text{val}} = \min(n_{\text{val}}, M_{\min} - n'_{\text{train}})`$, which reproduces the v1.5.0 batches.

For every class $`c \in \mathcal{C}`$, a subset $`S_c \subseteq \mathcal{I}^{(c)}`$ of size $`n'_{\text{train}} + n'_{\text{pool}}`$ is drawn uniformly without replacement and randomly partitioned into disjoint parts $`S_{c,T}`$ ($`|S_{c,T}| = n'_{\text{train}}`$), $`S_{c,\text{sel}}`$ ($`|S_{c,\text{sel}}| = n'_{\text{sel}}`$) and $`S_{c,\text{sc}}`$ ($`|S_{c,\text{sc}}| = n'_{\text{sc}}`$). The tree's training, selection and scoring sets are then:

$$
\mathcal{I}_ {T} = \bigcup_{c \in \mathcal{C}} S_{c,T}, \qquad \mathcal{I}_ {\text{sel}} = \bigcup_{c \in \mathcal{C}} S_{c,\text{sel}}, \qquad \mathcal{I}_ {\text{sc}} = \bigcup_{c \in \mathcal{C}} S_{c,\text{sc}}, \qquad \mathcal{I}_ {V} = \mathcal{I}_ {\text{sel}} \cup \mathcal{I}_ {\text{sc}},
$$

all pairwise disjoint. The selection set drives acceptance and Monte Carlo selection (§6.2); the scoring set is used only for the reported survival score, tree weights and feature importance (§6.1). When $`\gamma = 0`$, $`\mathcal{I}_{\text{sc}} = \emptyset`$ and $`\mathcal{I}_{\text{sel}} = \mathcal{I}_V`$. When `selection="ensemble"` (§6.4), the rows of a global ensemble holdout are removed from every $`\mathcal{I}^{(c)}`$ before these draws.

### Theoretical Properties of Balanced Micro-Batching

**1. Exact Class-Prior Uniformity.** Regardless of the global class frequencies $`\{M_c / M\}_{c \in \mathcal{C}}`$ in $`\mathcal{D}`$, $`\mathcal{I}_T`$, $`\mathcal{I}_{\text{sel}}`$ and $`\mathcal{I}_{\text{sc}}`$ (and hence $`\mathcal{I}_V`$) satisfy:

$$
\mathbb{P}_ {(x, y) \sim \mathcal{I}_ {T}}(y = c) = \mathbb{P}_ {(x, y) \sim \mathcal{I}_ {V}}(y = c) = \frac{1}{K}, \qquad \forall c \in \mathcal{C}.
$$

Tree *training* (split selection and leaf frequencies) is therefore structurally invariant to dataset imbalance. The output probabilities are not: leaf frequencies estimated on balanced batches are posteriors under a *uniform* class prior, and differ from the posteriors under the deployment prior unless they are corrected (§7.2).

**2. Equivalence of Validation Accuracy to Macro-Recall.** Because $`|\mathcal{I}_V \cap \mathcal{I}^{(c)}| = n'_{\text{val}}`$ is identical across all $`c \in \mathcal{C}`$ (and likewise for $`\mathcal{I}_{\text{sel}}`$ and $`\mathcal{I}_{\text{sc}}`$ separately), the empirical validation accuracy of a tree $`T`$ over $`\mathcal{I}_V`$,

$$
s(T; \mathcal{I}_ {V}) = \frac{1}{K\, n'_ {\text{val}}} \sum_{c \in \mathcal{C}} \sum_{i \in S_{c,V}} \mathbb{I}\!\big(\hat{y}_ {T}(\mathbf{x}_ {i}) = c\big) = \frac{1}{K} \sum_{c \in \mathcal{C}} \widehat{\text{Recall}}_ {c}(T; \mathcal{I}_ {V}),
$$

is identically equal to the **balanced accuracy** (macro-averaged per-class recall) on the validation batch. A tree that collapses onto a single class achieves exactly $`1/K`$ on $`s(T; \mathcal{I}_V)`$; in v1.6.0 such a tree never passes the threshold floor $`\tau_{\min} = 1/K + \varepsilon`$ (§6.2) and can be retained only as a last resort at the attempt cap, whereas in v1.5.0 it could survive through the best-of-$`R_{\max}`$ fallback or threshold decay.

**3. Dataset-Size Independence of Tree Induction.** Because $`|\mathcal{I}_T| = K \cdot n'_{\text{train}}`$ depends solely on the number of classes $`K`$ and batch parameter $`n_{\text{train}}`$, single-tree induction cost is independent of the full corpus size $`M`$ once class index buckets $`\{\mathcal{I}^{(c)}\}`$ are formed (unless the optional growth $`\rho > 0`$ is enabled, in which case $`n'_{\text{train}}`$ grows with $`M_{\min}`$ up to $`n_{\text{cap}}`$).

---

## 4. Stochastic Feature Subspace and Permutation Sampling

Before growing a candidate tree, RF-TSMC samples both the *cardinality* and the *ordered sequence* of predictor features:

**1. Cardinality Draw.** Draw $`k`$ uniformly from the discrete integer interval:

$$
k \sim \mathcal{U}\{f_{\min}, f_{\min}+1, \dots, f_{\max}\},
$$

clamped to $`1 \le f_{\min} \le f_{\max} \le d`$. With `feature_sampling="effective_depth"`, the upper end is truncated at the effective depth $`L_{\text{eff}}`$ (§5.4):

$$
k \sim \mathcal{U}\big\{f_{\min}, \dots, \max\!\big(f_{\min}, \min(f_{\max}, L_{\text{eff}})\big)\big\}.
$$

The default remains `feature_sampling="uniform"`.

**2. Ordered Sequence Draw.** Draw an ordered tuple of $`k`$ distinct features uniformly without replacement from the space of $`k`$-permutations of $`\mathcal{F}`$:

$$
F = (f^{(0)}, f^{(1)}, \dots, f^{(k-1)}) \sim \mathcal{U}\!\left(\mathcal{P}_ {k}(\mathcal{F})\right), \qquad |\mathcal{P}_ {k}(\mathcal{F})| = \frac{d!}{(d-k)!}.
$$

### Causal / Temporal Ordering Mode ($`\mathbb{I}_{\text{temp}} = \text{True}`$)

When predictor attributes represent indexed observations (e.g., lagged features `feat_1`, `feat_2`, $`\dots`$, `feat_T` with integer suffixes $`\tau(f) \in \mathbb{Z}_{\ge 0}`$), enabling temporal feature ordering sorts the sampled tuple $`F`$ in ascending order of $`\tau(f)`$:

$$
\tau(f^{(0)}) \le \tau(f^{(1)}) \le \dots \le \tau(f^{(k-1)}).
$$

Because tree induction processes $`F`$ sequentially from root to leaves, sorting $`F`$ enforces a structural precedence in which earlier temporal states condition the upper levels of the decision tree and later states refine deeper branches.

### Contrast with Classical Subspace Randomization

In Breiman's Random Forest, the subset cardinality $`m_{\text{try}}`$ is fixed globally and candidate features are resampled at *each internal node*, where a greedy impurity optimizer determines which feature sits at the root. In RF-TSMC, both the subset size $`k`$ and the **hierarchical split order** $`F`$ are stochastic, tree-level latent variables. Consequently, different candidate trees explore distinct low-dimensional subspaces ($`k \ll d`$) as well as broader feature combinations ($`k \approx d`$), and the root split feature $`f^{(0)}`$ is directly governed by the sampled permutation rather than a greedy global criterion.

### Effective Search Space and the Role of $`k`$

Because the queue is consumed one feature per tree level (§5.4) and median halving reaches pure leaves after about $`L_{\text{eff}} = \lceil \log_2(K\, n'_{\text{train}}) \rceil + 1`$ levels ($`\approx 5`$–$`6`$ with the defaults), the positions $`f^{(j)}`$ with $`j \ge L_{\text{eff}}`$ are almost never reached. Two candidates whose sequences share the same prefix of length $`L_{\text{eff}}`$ therefore produce (almost always) the same tree, and the Monte Carlo search is effectively over **ordered prefixes of length $`\approx L_{\text{eff}}`$**, not over $`\bigcup_k \mathcal{P}_k(\mathcal{F})`$. In particular, drawing $`k`$ uniformly from $`[f_{\min}, f_{\max}]`$ has little effect once $`k > L_{\text{eff}}`$: the extra positions are unused tail. The cardinality matters only when $`k < L_{\text{eff}}`$, in which case the cyclic queue wraps around and the same features are reused at deeper levels with new thresholds. The `"effective_depth"` law concentrates the draws on this informative range.

---

## 5. Impurity-Free Tree Induction via Cyclic Feature Queues

Given training index set $`\mathcal{I}_T`$ and ordered feature sequence $`F`$, a candidate decision tree $`T = \text{PlantTree}(\mathcal{I}_T, F)`$ is grown recursively from depth $`t = 1`$ without computing Gini impurity, entropy, or any target-dependent split scan (in the default configuration `split_supervision=0`; optional label-aware variants are described in §5.5).

![Impurity-Free Cyclic-Queue Tree Induction Flowchart](rf_tsmc_tree_induction.png)

*Figure 1: Impurity-free recursive tree induction (`PlantTree` / `GrowTree`) using deterministic median/mode splitting (`SplitData`) and cyclic feature-queue rotation. The split-operator box depicts the v1.5.0 cases (tie fallback, two-sample rule), which are retained only under `split_rule="legacy"`; the v1.6.0 default is the distinct-value midpoint rule of §5.1.*

### 5.1 Deterministic Split Operator $`\Phi(f, \mathcal{I})`$

At an internal node containing sample indices $`\mathcal{I}`$ ($`|\mathcal{I}| \ge 2`$ with at least two distinct classes) and candidate feature $`f`$, let $`v_i = x_{i, f}`$ for $`i \in \mathcal{I}`$ and $`n = |\mathcal{I}|`$. The deterministic partitioning operator $`\Phi(f, \mathcal{I}) \mapsto (\mathcal{I}_{\ge}, \mathcal{I}_{<}, \theta)`$ either returns a valid split (both children non-empty) or reports that $`f`$ yields **no valid split** at this node, in which case the queue rotates to the next feature (§5.2). The default rule (`split_rule="midpoint"`) proceeds by cases:

#### Case 1: $`f \in \mathcal{F}_{\text{num}}`$ (Numeric Distinct-Value Midpoint)
Let $`v_{(1)} \le v_{(2)} \le \dots \le v_{(n)}`$ denote the order statistics of $`\{v_i\}_{i \in \mathcal{I}}`$. The admissible cut positions are those lying between two adjacent *distinct* values,

$$
\mathcal{J} = \big\{ j \in \{1, \dots, n-1\} : v_{(j)} < v_{(j+1)} \big\},
$$

where a cut at $`j`$ sends exactly $`j`$ samples to the lower child. If $`\mathcal{J} = \emptyset`$ (the feature has a single distinct value on $`\mathcal{I}`$), no valid split exists. Otherwise the operator chooses the cut whose lower-child count is closest to the count-median, breaking ties toward the lower cut,

$$
j^{\ast} = \min \arg\min_{j \in \mathcal{J}} \big| j - n/2 \big|, \qquad \theta = \tfrac{1}{2}\big(v_{(j^{\ast})} + v_{(j^{\ast}+1)}\big),
$$

with the safeguard $`\theta \leftarrow v_{(j^{\ast}+1)}`$ if floating-point rounding yields $`\theta \le v_{(j^{\ast})}`$. The partition is exactly

$$
\mathcal{I}_ {\ge} = \{i \in \mathcal{I} : v_i \ge \theta\}, \qquad \mathcal{I}_ {<} = \{i \in \mathcal{I} : v_i < \theta\},
$$

with $`|\mathcal{I}_{<}| = j^{\ast} \ge 1`$ and $`|\mathcal{I}_{\ge}| = n - j^{\ast} \ge 1`$. Because $`v_{(j^{\ast})} < \theta \le v_{(j^{\ast}+1)}`$ and no sample value lies strictly between them, the inference rule $`x_f \ge \theta`$ (§7.1) **reproduces the training partition exactly**: every training row reaches the leaf it was trained into, and every leaf is reachable. Without ties, $`j^{\ast} = \lfloor n/2 \rfloor`$ and the split is the usual median bisection; with ties, the cut moves to the nearest boundary between distinct values instead of producing an empty child. The two-sample case ($`n = 2`$) needs no special treatment: distinct values give two singletons separated at their midpoint, and equal values give no valid split.

#### Case 2: $`f \in \mathcal{F}_{\text{cat}}`$ (Categorical Modal Partition)
The split value $`\theta`$ is the statistical mode (most frequent category) in $`\mathcal{I}`$, with ties broken deterministically by the smallest category value:

$$
\theta = \min \arg\max_{u} \sum_{i \in \mathcal{I}} \mathbb{I}(v_i = u).
$$

The node induces a binary one-vs.-rest categorical split:

$$
\mathcal{I}_ {\ge} = \{i \in \mathcal{I} : v_i = \theta\}, \qquad \mathcal{I}_ {<} = \{i \in \mathcal{I} : v_i \neq \theta\}.
$$

If only one category is present in $`\mathcal{I}`$, then $`\mathcal{I}_{<} = \emptyset`$ and $`f`$ yields no valid split. (An `"ordered"` set-valued alternative is described in §5.5.)

#### Stored Node Counts
Every split node stores, together with $`(f, \theta)`$, the training counts $`n_{\ge} = |\mathcal{I}_{\ge}|`$ and $`n_{<} = |\mathcal{I}_{<}|`$, and every leaf stores its training count $`n_{\ell}`$. These counts are used by the weighted missing-value rule (§7.1) and by leaf smoothing (§5.2, §7.2).

#### Legacy Rule (`split_rule="legacy"`, v1.5.0)
For reproducibility, the v1.5.0 operator is retained under `split_rule="legacy"`. For $`n > 2`$ numeric features it sets $`\theta = \text{median}(\{v_i\})`$ and partitions by $`v_i \ge \theta`$; whenever this leaves a child empty (e.g., $`\theta = \min_i v_i`$ under ties), it switches to the strict partition $`\{v_i > \theta\}`$ / $`\{v_i \le \theta\}`$. For $`n = 2`$ it sets $`\theta = v_{(2)}`$ and returns the singletons $`\{i_{(2)}\}`$ / $`\{i_{(1)}\}`$. Both rules are inconsistent with the inference rule $`x_f \ge \theta`$, which is unchanged:

- *Tie fallback:* rows with $`v_i = \theta`$ are trained into the $`<`$ child (because $`v_i \le \theta`$) but are routed to the $`\ge`$ branch at inference. Training rows, and any future query equal to $`\theta`$, therefore land in a leaf that never saw them.
- *Two-sample rule:* if $`v_{(1)} = v_{(2)}`$, both samples satisfy $`x_f \ge \theta`$ at inference, so the $`<`$ leaf, which holds a sample of the other class, is **unreachable**.

The midpoint rule eliminates both defects by construction.

### 5.2 Cyclic Queue Traversal and Leaf Generation

To enforce feature diversity down every root-to-leaf path without exhausting features prematurely on deep trees, `GrowTree` maintains $`F`$ as a **cyclic FIFO queue**:

1. **Base Termination Conditions:** A leaf node $`\ell`$ is returned immediately if:
   - Depth reaches the cap: $`t \ge d_{\max}`$, or
   - Node purity is achieved: $`|\{y_i : i \in \mathcal{I}\}| = 1`$.
2. **Cyclic Feature Selection:** Let $`f_{\text{start}} = F[0]`$ be the initial head of the queue. In a loop:
   - Pop the head feature $`f \leftarrow F[0]`$ and compute $`(\mathcal{I}_{\ge}, \mathcal{I}_{<}, \theta) = \Phi(f, \mathcal{I})`$.
   - Rotate $`f`$ to the tail of the queue: $`F \leftarrow (F[1], \dots, F[k-1], f)`$.
   - **Admissibility Check:** Accept $`(f, \theta)`$ if and only if $`\Phi`$ returned a valid split and both children satisfy the minimum partition cardinality:

$$
|\mathcal{I}_ {\ge}| \ge s_{\min} \quad \text{and} \quad |\mathcal{I}_ {<}| \ge s_{\min}.
$$

   - If admissible, emit internal node $`(f, \theta)`$ and recurse independently on $`(\mathcal{I}_{\ge}, F, t+1)`$ and $`(\mathcal{I}_{<}, F, t+1)`$, each receiving a fresh copy of the rotated queue $`F`$.
   - If not admissible and the queue completes a full rotation ($`F[0] = f_{\text{start}}`$) without any feature producing an admissible split, terminate recursion and return a leaf node $`\ell`$.
3. **Leaf Empirical Distribution:** Each leaf $`\ell`$ at depth $`t`$ records its training count $`n_{\ell} = |\mathcal{I}|`$ and the class counts $`n_{\ell}(c) = \sum_{i \in \mathcal{I}} \mathbb{I}(y_i = c)`$, from which the leaf distribution is computed with Laplace smoothing parameter $`\alpha`$ = `leaf_alpha` $`\ge 0`$:

$$
p_{\ell}(c) = \frac{n_{\ell}(c) + \alpha}{n_{\ell} + K\alpha}, \qquad c \in \mathcal{C}.
$$

   The default $`\alpha = 0`$ gives the empirical frequencies $`p_{\ell}(c) = n_{\ell}(c)/n_{\ell}`$ of v1.5.0 (see §7.2 for the motivation of $`\alpha > 0`$).

### 5.3 Formal Algorithm for Tree Induction

```text
Algorithm 1: Impurity-Free Cyclic-Queue Tree Induction (PlantTree)
Input : Training index set I_T, ordered feature queue F, depth bound d_max, min child size s_min
Output: Decision tree T
(Shown for split_supervision in {0, 1}; level 2 replaces the repeat-loop, see Section 5.5.)

function GrowTree(I, F, t):
    if I is empty or F is empty or t >= d_max or |{y_i : i in I}| == 1 then
        return Leaf(n = |I|, counts n(c) = sum_{i in I} 1[y_i == c], depth = t)
    end if

    f_start <- F[0]
    repeat
        f <- F[0]
        (valid, I_ge, I_lt, theta) <- SplitData(f, I)   // Midpoint rule (Section 5.1)
        F <- RotateLeft(F)                    // Move f from front to back
        if valid and |I_ge| >= s_min and |I_lt| >= s_min then
            child_ge <- GrowTree(I_ge, Copy(F), t + 1)
            child_lt <- GrowTree(I_lt, Copy(F), t + 1)
            return SplitNode(feature = f, threshold = theta,
                             n_ge = |I_ge|, n_lt = |I_lt|,
                             branch_ge = child_ge, branch_lt = child_lt)
        end if
    until F[0] == f_start

    return Leaf(n = |I|, counts n(c) = sum_{i in I} 1[y_i == c], depth = t)
end function
```

### 5.4 Theoretical Insights on Tree Structure

- **Logarithmic Depth on Continuous Numeric Features:** When the features used are continuous numeric features without ties on the node, each midpoint split bisects $`|\mathcal{I}|`$ into two subsets of sizes $`\lfloor |\mathcal{I}|/2 \rfloor`$ and $`\lceil |\mathcal{I}|/2 \rceil`$. Consequently, a tree grown to pure leaves on $`|\mathcal{I}_T| = K \cdot n'_{\text{train}}`$ samples has depth bounded by the **effective depth** $`L_{\text{eff}} = \lceil \log_2(K \cdot n'_{\text{train}}) \rceil + 1`$ ($`L_{\text{eff}} = 6`$ for $`K \in \{2, 3\}`$ and $`n'_{\text{train}} = 10`$). This bound does **not** hold in general: ties move the cut away from the count-median, and categorical mode-vs.-rest splits peel off one category per level, so trees using low-cardinality or skewed categorical features can form chains deeper than $`L_{\text{eff}}`$.
- **One Feature per Level (Feature-Oblivious Levels):** Both children of a split receive the same rotated copy of the queue (§5.2), and with $`s_{\min} = 1`$ a midpoint split fails only when the feature is constant on the node. Hence, whenever no split fails, every node at depth $`t`$ splits on the same feature $`F[(t-1) \bmod k]`$: the tree is *oblivious in its feature choice* (as in oblivious decision trees, e.g., CatBoost; Prokhorenkova et al., 2018), though not in its thresholds, which are node-specific medians. Branches diverge in feature order only where a split fails and the queue of that subtree rotates further. Together with the previous point, positions of $`F`$ beyond $`L_{\text{eff}}`$ are almost never used (§4).
- **Controlled Over-Specialization:** Growing unpruned trees ($`d_{\max} = \infty, s_{\min} = 1`$) on a small balanced subset $`\mathcal{I}_T`$ using a subset $`F`$ causes each individual tree $`T`$ to strongly fit the geometric partitioning induced by $`(F, \mathcal{I}_T)`$. Unlike classical decision trees where overfitting on the full dataset increases variance across correlated splits, here each tree specializes in a *different* low-dimensional projection $`F`$ and micro-batch $`\mathcal{I}_T`$, and must immediately prove non-trivial generalization on the disjoint validation set $`\mathcal{I}_V`$.

### 5.5 Optional Label-Aware Splits

The default constructor is label-free apart from the purity stopping rule. Two opt-in mechanisms inject a limited amount of label information while preserving the midpoint structure of §5.1. For a candidate partition, let the weighted Gini impurity be

$$
G(\mathcal{I}_ {\ge}, \mathcal{I}_ {<}) = \frac{|\mathcal{I}_ {\ge}|}{|\mathcal{I}|}\,\text{Gini}(\mathcal{I}_ {\ge}) + \frac{|\mathcal{I}_ {<}|}{|\mathcal{I}|}\,\text{Gini}(\mathcal{I}_ {<}), \qquad \text{Gini}(\mathcal{J}) = 1 - \sum_{c \in \mathcal{C}} \Big(\tfrac{1}{|\mathcal{J}|} \textstyle\sum_{i \in \mathcal{J}} \mathbb{I}(y_i = c)\Big)^2.
$$

**Supervision levels (`split_supervision` $`= \lambda_{\text{sup}}`$).**
- $`\lambda_{\text{sup}} = 0`$ (default): the count-median cut of §5.1.
- $`\lambda_{\text{sup}} = 1`$: for the current feature, the operator evaluates the distinct-value cuts closest to the 25th, 50th and 75th count-percentiles,
  $`j_q = \min \arg\min_{j \in \mathcal{J}} |j - q\,n|`$ for $`q \in \{1/2, 1/4, 3/4\}`$, each with its midpoint threshold, and keeps the one with the lowest $`G`$ (the median cut wins ties). The partition and routing rule are those of §5.1, so the training partition is still reproduced exactly at inference.
- $`\lambda_{\text{sup}} = 2`$: in addition, at each node the first $`H`$ = `split_lookahead` (default $`2`$) features of the queue are evaluated with the level-1 operator (the scan continues further down the queue only until at least one valid split is found), and the valid split with the lowest $`G`$ is kept (earlier queue position wins ties). The queue is then rotated to the position immediately after the chosen feature, $`F \leftarrow (F[p+1], \dots, F[k-1], F[0], \dots, F[p])`$ for chosen position $`p`$, and both children receive this rotated queue. This weakens the one-feature-per-level property of §5.4 in exchange for a short greedy look-ahead.

**Ordered categorical set splits (`categorical_split="ordered"`).** Instead of mode-vs.-rest, the categories $`u`$ present at the node are sorted by the empirical class share

$$
\hat{P}(y = c^{\ast} \mid v = u) = \frac{\sum_{i \in \mathcal{I}} \mathbb{I}(v_i = u,\, y_i = c^{\ast})}{\sum_{i \in \mathcal{I}} \mathbb{I}(v_i = u)},
$$

where $`c^{\ast}`$ is the class with index $`1`$ for binary targets and the node's majority class otherwise (ties in the share are broken by category value). The rows, ordered by the rank of their category, are cut between two adjacent distinct ranks at the count-median (or at the best of the percentile cuts when $`\lambda_{\text{sup}} \ge 1`$). This yields a **set split**: the node has `feat_type` `categorical_set` and stores as `split_val` the set $`\Theta`$ of categories placed above the cut, with

$$
\mathcal{I}_ {\ge} = \{i \in \mathcal{I} : v_i \in \Theta\}, \qquad \mathcal{I}_ {<} = \{i \in \mathcal{I} : v_i \notin \Theta\}.
$$

At inference the test is set membership $`x_f \in \Theta`$; categories not seen at the node (including categories unseen in training) go to the $`<`$ branch. A node with a single category yields no valid split. For binary targets, ordering the $`m`$ categories by class share is the standard device that reduces the $`2^{m-1} - 1`$ candidate category subsets to $`m - 1`$ ordered cuts; here the ordering is cut at count percentiles rather than searched exhaustively.

---

## 6. Monte Carlo Tree Selection

Because `PlantTree` executes no target-informed feature or split optimization internally, an unvalidated tree constructed from an uninformative or poorly ordered feature sequence $`F`$ may perform poorly out of sample. RF-TSMC resolves this through **Monte Carlo rejection-sampling selection** over $`(k, F)`$ conditioned on validation performance.

### 6.1 Validation Scoring Function

For each candidate tree $`T`$ planted on $`\mathcal{I}_T`$, let $`\hat{y}_T(\mathbf{x}) = \arg\max_{c \in \mathcal{C}} p_T(c \mid \mathbf{x})`$ be its hard class prediction (where $`p_T(c \mid \mathbf{x})`$ is defined in Section 7.1). The accuracy of $`T`$ on a balanced index set $`\mathcal{J}`$ is:

$$
s(T; \mathcal{J}) = \frac{1}{|\mathcal{J}|} \sum_{i \in \mathcal{J}} \mathbb{I}\!\big(\hat{y}_ {T}(\mathbf{x}_ {i}) = y_i\big) \in [0, 1].
$$

RF-TSMC computes two scores per candidate: the **selection score** $`s_{\text{sel}} = s(T; \mathcal{I}_{\text{sel}})`$, which alone drives acceptance and Monte Carlo selection (§6.2), and, when $`\gamma > 0`$, the **scoring score** $`s_{\text{sc}} = s(T; \mathcal{I}_{\text{sc}})`$. The reported survival score of an accepted tree is

$$
s_{\text{surv}}(T) = \begin{cases}
  s(T; \mathcal{I}_ {\text{sc}}), & \mathcal{I}_ {\text{sc}} \neq \emptyset, \\
  s(T; \mathcal{I}_ {\text{sel}}), & \mathcal{I}_ {\text{sc}} = \emptyset \;(\gamma = 0,\ \text{v1.5.0 behavior}),
\end{cases}
$$

and $`s_{\text{surv}}`$ is the quantity used for tree weights (§7.2) and survival-based feature importance (§8.1). When $`\alpha > 0`$, the Brier score $`\frac{1}{|\mathcal{I}_{\text{sel}}|}\sum_{i} \sum_{c} \big(p_T(c \mid \mathbf{x}_i) - \mathbb{I}(y_i = c)\big)^2`$ on the selection set is also computed and used to break ties in $`s_{\text{sel}}`$. (In the degenerate boundary case where $`M_{\min} \le n_{\text{train}}`$ leaves $`\mathcal{I}_{\text{sel}} = \emptyset`$, the selection score defaults to $`1.0`$.)

#### Selection Bias on Small Validation Sets
With the defaults ($`n_{\text{val}} = 10`$, $`K = 2`$) the selection set has only $`20`$ rows. For a tree of true balanced accuracy $`0.75`$, the binomial standard deviation of $`s_{\text{sel}}`$ is $`\sqrt{0.75 \cdot 0.25 / 20} \approx 0.10`$, and scores are quantized in steps of $`0.05`$. Choosing the best of $`R_{\max} = 10`$ equally good candidates inflates the winner's score by the expected maximum of $`10`$ standard normal variables, $`\approx 1.54`$ standard deviations, i.e., by $`\approx 0.15`$ from noise alone. Consequently:
- a survival score computed on the same set used for selection is **optimistically biased**, which is why $`\gamma > 0`$ reports $`s_{\text{surv}}`$ on the independent scoring set;
- the default $`\tau_0 = 1.0`$ mostly rewards luck on $`20`$ rows rather than intrinsically better trees;
- score ties are frequent, so the identity of the selected tree within a tie class is arbitrary (hence the Brier tie-break when $`\alpha > 0`$).

Enlarging the pool through `val_pool_pclass` reduces this variance at the cost of validation time (§10).

### 6.2 Dynamic Acceptance–Rejection Regimes

**Threshold floor and attempt cap.** v1.6.0 introduces an above-chance floor on the acceptance threshold,

$$
\tau_{\min} = \frac{1}{K} + \varepsilon,
$$

with $`\varepsilon`$ = `tau_floor_eps` (default $`0.05`$), and a cap $`A_{\max}`$ = `max_tree_attempts` on the number of candidates per tree slot (`"auto"`: $`A_{\max} = 10\,R_{\max}`$ when the floor is active, unlimited otherwise). Setting $`\varepsilon`$ = None recovers the v1.5.0 rules ($`\tau_{\min} = -\infty`$, no cap).

For tree slot $`b`$, a batch triple $`(\mathcal{I}_T, \mathcal{I}_{\text{sel}}, \mathcal{I}_{\text{sc}})`$ is drawn (§3) and the selection loop initializes acceptance threshold $`\tau \leftarrow \tau_0`$, rejection counter $`r \leftarrow 0`$, attempt counter $`a \leftarrow 0`$, batch-best tree $`T^* \leftarrow \varnothing`$ with key $`\kappa^* \leftarrow -\infty`$ (strictly below any attainable key, so the first candidate is always recorded), and slot-best tree $`T^{\dagger} \leftarrow \varnothing`$ with key $`\kappa^{\dagger} \leftarrow -\infty`$. The comparison key of a candidate is $`\kappa(T) = s_{\text{sel}}`$, refined lexicographically to $`\kappa(T) = (s_{\text{sel}}, -\text{Brier}_{\text{sel}})`$ when $`\alpha > 0`$. At each Monte Carlo iteration:

1. Draw a fresh random feature sequence $`F \sim \text{SampleFeats}(\mathcal{F})`$ and set $`a \leftarrow a + 1`$.
2. Induce candidate tree $`T \leftarrow \text{PlantTree}(\mathcal{I}_T, F)`$.
3. Evaluate $`s_{\text{sel}} = s(T; \mathcal{I}_{\text{sel}})`$ (and $`s_{\text{sc}}`$ if $`\mathcal{I}_{\text{sc}} \neq \emptyset`$); update $`(T^*, \kappa^*)`$ and $`(T^{\dagger}, \kappa^{\dagger})`$ if $`\kappa(T)`$ exceeds them.
4. **Acceptance Check:** If $`s_{\text{sel}} \ge \tau`$ **and** $`s_{\text{sel}} \ge \tau_{\min}`$, accept $`T`$ immediately with survival score $`s_{\text{surv}}(T)`$ (§6.1) and terminate.
5. **Attempt Cap:** Otherwise increment $`r \leftarrow r + 1`$. If $`a \ge A_{\max}`$, return the slot-best candidate $`T^{\dagger}`$ over all batches drawn for the slot, flag it `hit_attempt_cap`, and terminate; the library emits a warning reporting how many slots hit the cap.
6. **Rejection Handling:** Two regimes govern budget exhaustion ($`r \ge R_{\max}`$):

**Regime A — Bounded-Patience Best-of-$`R_{\max}`$ Selection ($`\beta_{\text{best}} = \text{True}`$).** During the $`R_{\max}`$ trials on the current batch, maintain the incumbent optimum:

$$
(T^{\ast}, \kappa^{\ast}) \leftarrow \begin{cases}
  (T, \kappa(T)), & \text{if } \kappa(T) > \kappa^{\ast}, \\
  (T^{\ast}, \kappa^{\ast}), & \text{otherwise.}
\end{cases}
$$

If $`r`$ reaches $`R_{\max}`$ without an acceptance, return $`T^*`$ **provided its selection score satisfies $`s_{\text{sel}}(T^*) \ge \tau_{\min}`$**. Formally, if $`T^{(1)}, \dots, T^{(R_{\max})}`$ fail the threshold, this implements:

$$
T_ {\text{surv}} = \arg\max_{1 \le j \le R_ {\max}} \kappa\big(T^{(j)}\big) \quad \text{if } \max_{j} s\big(T^{(j)}; \mathcal{I}_ {\text{sel}}\big) \ge \tau_{\min}.
$$

Otherwise the batch is judged unable to support an above-chance tree: a new triple $`(\mathcal{I}_T, \mathcal{I}_{\text{sel}}, \mathcal{I}_{\text{sc}})`$ is drawn and $`\tau \leftarrow \tau_0`$, $`r \leftarrow 0`$, $`T^* \leftarrow \varnothing`$, $`\kappa^* \leftarrow -\infty`$ are reset (the slot-best $`T^{\dagger}`$ and the attempt counter $`a`$ are kept). Thus, selection stops early whenever an exceptional tree achieves $`s_{\text{sel}} \ge \tau_0`$, and otherwise performs bounded random search of budget $`R_{\max}`$ per batch over the effective prefix space of §4.

**Regime B — Adaptive Threshold Relaxation ($`\beta_{\text{best}} = \text{False}`$).** Once the patience counter reaches $`r \ge R_{\max}`$, subsequent rejections relax the acceptance threshold linearly on every unsuccessful trial:

$$
\tau \leftarrow \tau - \Delta\tau,
$$

while continuing to sample fresh candidate sequences $`F`$. If the relaxation would take $`\tau`$ below the floor ($`\tau - \Delta\tau < \tau_{\min}`$), the threshold is not lowered; instead a new batch triple is drawn and $`\tau \leftarrow \tau_0`$, $`r \leftarrow 0`$ are reset. The threshold therefore never falls below $`\tau_{\min}`$, and termination is guaranteed by the attempt cap $`A_{\max}`$ rather than by $`\tau`$ reaching $`0`$. (In legacy mode, $`\tau_{\min} = -\infty`$ and $`A_{\max} = \infty`$, and termination follows, as in v1.5.0, from $`s_{\text{sel}} \ge 0`$ and the monotone decrease of $`\tau`$.)

```text
Algorithm 2: Monte Carlo Tree Selection (SurvivedTree) and Forest Assembly
Input : Dataset D, forest size B, threshold tau_0, step delta_th, patience R_max, mode beta_best,
        floor tau_min = 1/K + eps (-inf if eps = None), attempt cap A_max (inf if None)
Output: Ensemble E = {(T_1, s_1), ..., (T_B, s_B)}

E <- empty list
for b = 1 to B do                               // Embarrassingly parallel across b
    (I_T, I_sel, I_sc) <- SplitTrainVal(D)      // Class-balanced disjoint batches (Section 3)
    tau <- tau_0;  r <- 0;  a <- 0
    T* <- None;  key* <- -inf                   // Batch best: first candidate always recorded
    T+ <- None;  key+ <- -inf                   // Slot best over all batches

    loop
        F <- SampleFeats(F_all);  a <- a + 1    // Random size k and permutation
        T <- PlantTree(I_T, F)
        s_sel <- Validate(T, I_sel)
        key <- (s_sel, -Brier(T, I_sel)) if alpha > 0 else s_sel
        if key > key* then T* <- T;  key* <- key end if
        if key > key+ then T+ <- T;  key+ <- key end if

        if s_sel >= tau and s_sel >= tau_min then
            T_surv <- T;  break                 // Accepted
        end if

        r <- r + 1
        if a >= A_max then
            T_surv <- T+;  flag T_surv.hit_attempt_cap;  break
        end if
        if r >= R_max then
            if beta_best then
                if s_sel(T*) >= tau_min then
                    T_surv <- T*;  break
                else                            // Batch cannot reach the floor
                    (I_T, I_sel, I_sc) <- SplitTrainVal(D)
                    tau <- tau_0;  r <- 0;  T* <- None;  key* <- -inf
                end if
            else if tau - delta_th < tau_min then
                (I_T, I_sel, I_sc) <- SplitTrainVal(D)
                tau <- tau_0;  r <- 0;  T* <- None;  key* <- -inf
            else
                tau <- tau - delta_th
            end if
        end if
    end loop

    s_surv <- s_sc(T_surv) if I_sc nonempty else s_sel(T_surv)   // On T_surv's own batch
    Record used feature subset U(T_surv) <= F
    Append (T_surv, s_surv) to E
end for
if any tree flagged hit_attempt_cap then warn
return E
```

### 6.3 Probabilistic Interpretation of Survival Selection

Fix a batch draw $`(\mathcal{I}_T, \mathcal{I}_{\text{sel}})`$, and let $`\mu_{\mathcal{F}}`$ denote the sampling distribution over feature sequences $`F`$. Define the **survival probability** at threshold $`\tau`$ as:

$$
q(\tau \mid \mathcal{I}_ {T}, \mathcal{I}_ {\text{sel}}) = \mathbb{P}_ {F \sim \mu_{\mathcal{F}}}\!\big(s(T(\mathcal{I}_ {T}, F); \mathcal{I}_ {\text{sel}}) \ge \max(\tau, \tau_{\min})\big).
$$

- Under **Regime A** ($`\beta_{\text{best}} = \text{True}`$), the probability that a tree passes $`\tau_0`$ within $`R_{\max}`$ trials on one batch is $`1 - (1 - q(\tau_0))^{R_{\max}}`$, and the expected number of tree evaluations per batch is bounded by $`\sum_{j=0}^{R_{\max}-1} (1 - q(\tau_0))^j \le R_{\max}`$. When no trial meets $`\tau_0`$, the output score follows the maximum order statistic $`S_{(R_{\max})} = \max_{1 \le j \le R_{\max}} S_j`$, whose cumulative distribution function is $`F_{S_{(R_{\max})}}(s) = [F_S(s)]^{R_{\max}}`$; the batch is kept if $`S_{(R_{\max})} \ge \tau_{\min}`$, which happens with probability $`1 - [F_S(\tau_{\min}^-)]^{R_{\max}}`$, and is redrawn otherwise. Because $`S_j`$ is measured on the same small set used to choose the maximum, the shift of $`F_{S_{(R_{\max})}}`$ toward high scores reflects both genuinely better feature sequences and selection noise (§6.1).
- Under **Regime B** ($`\beta_{\text{best}} = \text{False}`$), for any threshold $`\tau_m = \tau_0 - m\,\Delta\tau \ge \tau_{\min}`$ reached after $`R_{\max} + m`$ failures ($`m \ge 0`$), the conditional survival probability increases monotonically with $`m`$ ($`q(\tau_{m+1}) \ge q(\tau_m)`$), but is capped at $`q(\tau_{\min})`$: the threshold no longer decays to $`\tau \le 0`$, a batch on which $`q(\tau_{\min})`$ is small is abandoned and redrawn, and termination is enforced by $`A_{\max}`$.
- **Implicit Subspace Filtering:** Because the batch is held fixed during the inner loop for slot $`b`$ (until a redraw), Monte Carlo selection compares candidate feature sequences $`F^{(1)}, F^{(2)}, \dots`$ on an *identical* train–selection benchmark, so differences in $`s_{\text{sel}}`$ are not confounded by batch-to-batch sampling variation. They are, however, measured on only $`K\, n'_{\text{sel}}`$ rows (20 with the defaults), so for candidates of similar quality they are dominated by binomial noise rather than by the predictive quality of $`F`$ (§6.1); larger selection pools or ensemble selection (§6.4) are needed for the comparison to be informative. Across the outer loop $`b = 1, \dots, B`$, resampling the batches decorrelates the trees across both data samples and feature subspaces.

### 6.4 Ensemble Selection over All Candidates (`selection="ensemble"`)

Per-slot selection (`selection="per_tree"`, the default) chooses each tree for its *individual* strength. Breiman's generalization bound for random forests depends on both the mean strength of the trees and their mean correlation; selecting each slot's tree for strength on similar data can raise correlation without a proportional gain in ensemble accuracy. As an alternative, `selection="ensemble"` assembles the forest by greedy ensemble selection (Caruana et al., 2004):

1. **Global holdout.** Before any slot is grown, a balanced holdout of $`n'_{\text{ens}} = \max\!\big(1, \min(n_{\text{ens}}, \lfloor M_{\min}/2 \rfloor)\big)`$ rows per class ($`n_{\text{ens}}`$ = `ensemble_pool_pclass`) is set aside and excluded from all slot batches. It is split by $`\gamma`$ into an ensemble-selection part $`\mathcal{H}_{\text{sel}}`$ and an ensemble-scoring part $`\mathcal{H}_{\text{sc}}`$.
2. **Candidate library.** The per-slot Monte Carlo loops of §6.2 run unchanged, but *every* candidate $`T_j`$ of every slot is kept, together with its class-probability matrix $`P_j \in [0, 1]^{|\mathcal{H}_{\text{sel}}| \times K}`$ on $`\mathcal{H}_{\text{sel}}`$.
3. **Greedy forward selection with replacement.** Let $`m(\cdot)`$ be the `ensemble_metric` (balanced accuracy, or mean log-probability of the true class for `"logloss"`) of a soft-voting ensemble, with plateaus of the discrete metric broken lexicographically by the mean log-probability of the true class. Start from the $`B_0`$ = `ensemble_init` (default $`3`$) individually best candidates, $`\mathcal{S} = (j_1, \dots, j_{B_0})`$, and repeat

   $$
   j^{+} = \arg\max_{j} \; m\!\Big(\frac{1}{|\mathcal{S}| + 1}\Big(\sum_{j' \in \mathcal{S}} P_{j'} + P_j\Big)\Big), \qquad \mathcal{S} \leftarrow \mathcal{S} \cup (j^{+}),
   $$

   until $`|\mathcal{S}| = B`$ or adding the best candidate would decrease the metric. Candidates may be selected more than once, which acts as an integer weighting.
4. **Reporting.** The forest is $`\{T_j : j \in \mathcal{S}\}`$ (with multiplicity), and its final ensemble score is reported on $`\mathcal{H}_{\text{sc}}`$ (on $`\mathcal{H}_{\text{sel}}`$ if $`\gamma = 0`$).

Because the objective is evaluated on the ensemble rather than on single trees, candidates that are individually weaker but make errors on different rows can be preferred, directly targeting the strength–correlation trade-off. The cost is the memory for $`|\text{library}| \times |\mathcal{H}_{\text{sel}}| \times K`$ probabilities and the removal of $`n'_{\text{ens}}`$ rows per class from tree training.

*Empirical status (v1.6.0 ablation, see `changelog.md`).* On the benchmark datasets this option has not beaten per-slot selection. With the balanced-accuracy metric, the greedy search stops early on a holdout of about 100 rows per class and keeps about 14 distinct trees, which overfits the holdout. The log-probability metric keeps the full forest and nearly closes the gap. Treat the option as experimental.

---

## 7. Inference, Missing-Value Replication, and Ensemble Aggregation

### 7.1 Single-Tree Prediction and Missing-Feature Branch Replication

Let $`\mathbf{x}`$ be a query observation (whose schema may omit features or contain `NaN` entries). Starting at the root of tree $`T`$, traversal proceeds recursively at each internal node testing feature $`f`$ with split value $`\theta`$:

**Observed Feature Value ($`x_f`$ present and non-missing).** Traversal follows the deterministic routing function:

$$
\text{next}(\mathbf{x}; f, \theta) = \begin{cases}
  \text{branch}_ {\ge}, & \text{if } (f \in \mathcal{F}_ {\text{num}} \land x_f \ge \theta) \;\lor\; (f \in \mathcal{F}_ {\text{cat}} \land x_f = \theta) \;\lor\; (\text{set split} \land x_f \in \Theta), \\
  \text{branch}_ {<}, & \text{otherwise.}
\end{cases}
$$

With the midpoint rule (§5.1), this routing reproduces the training partition exactly. A categorical value never seen at the node (including categories unseen in training) is **not** treated as missing: it fails the equality or membership test and is routed to $`\text{branch}_{<}`$.

**Missing or Omitted Feature Value ($`x_f`$ absent or `NaN`).** Instead of halting at an incomplete node, using surrogate splits, or requiring prior imputation, RF-TSMC applies **structural branch replication**: traversal forks simultaneously down *both* $`\text{branch}_{\ge}`$ and $`\text{branch}_{<}`$. Both engines treat an absent key and a `NaN` value identically. How the reached leaves are combined is set by `missing_strategy`.

**Weighted marginalization (`missing_strategy="weighted"`, default).** Following C4.5 (Quinlan, 1993), each branch is weighted by the fraction of training rows that took it. For a node $`\nu`$, a query $`\mathbf{x}`$ and an incoming path weight $`w`$, define recursively

$$
\text{LeafDist}(\nu, \mathbf{x}, w) = \begin{cases}
  w \cdot p_{\nu}(\cdot), & \nu \text{ is a leaf}, \\
  \text{LeafDist}\big(\text{next}(\mathbf{x}; f_\nu, \theta_\nu), \mathbf{x}, w\big), & x_{f_\nu} \text{ observed}, \\
  \text{LeafDist}\big(\nu_{\ge}, \mathbf{x}, w\,\tfrac{n_{\ge}}{n_{\ge} + n_{<}}\big) + \text{LeafDist}\big(\nu_{<}, \mathbf{x}, w\,\tfrac{n_{<}}{n_{\ge} + n_{<}}\big), & x_{f_\nu} \text{ missing},
\end{cases}
$$

where $`n_{\ge}, n_{<}`$ are the training counts stored at $`\nu`$ (§5.1); for v1.5.0 trees that lack stored counts, the fractions default to $`1/2`$ and $`1/2`$. The tree posterior is

$$
p_T(c \mid \mathbf{x}) = \text{LeafDist}(\text{root}(T), \mathbf{x}, 1)(c), \qquad c \in \mathcal{C}.
$$

The path weights of the reached leaves are non-negative and sum to $`1`$ (each missing split splits the incoming weight into two parts summing to it), so $`p_T(\cdot \mid \mathbf{x})`$ is a proper distribution without renormalization, and it equals the expectation of the leaf distribution under the training-conditional law of the missing split outcomes.

**Uniform leaf average (`missing_strategy="uniform_leaves"`, v1.5.0).** Recursively collecting all leaves reached by $`\mathbf{x}`$ yields a non-empty leaf multiset $`\mathcal{L}_T(\mathbf{x}) = \{\ell_1, \dots, \ell_m\}`$ (with $`m = 1`$ when all traversed features are observed, and $`m \le 2^{h}`$ when $`h`$ missing splits are encountered along the active paths). The legacy posterior is the normalized uniform average over all reachable leaves:

$$
\tilde{p}_ {T}(c \mid \mathbf{x}) = \frac{1}{|\mathcal{L}_ {T}(\mathbf{x})|} \sum_{\ell \in \mathcal{L}_ {T}(\mathbf{x})} p_{\ell}(c), \qquad p_T(c \mid \mathbf{x}) = \frac{\tilde{p}_ {T}(c \mid \mathbf{x})}{\sum_{c' \in \mathcal{C}} \tilde{p}_ {T}(c' \mid \mathbf{x})}, \qquad c \in \mathcal{C}.
$$

*Correction of the v1.5.0 interpretation:* the v1.5.0 specification described this rule as an "unweighted marginalization" over the missing split condition $`\mathbb{I}(x_f \ge \theta) \in \{0, 1\}`$. That holds only when both branches of every missing split reach the same number of leaves. Each leaf receives weight $`1/m`$, so a branch's total weight is proportional to its *number of reached leaves*, not to $`1/2`$. For example, if $`x_f`$ is missing at the root, the $`<`$ branch is a single leaf, and the $`\ge`$ branch reaches two leaves through a second missing split, the uniform rule assigns total weight $`2/3`$ to the $`\ge`$ side instead of the $`1/2`$ of a genuinely unweighted marginalization (and instead of $`n_{\ge}/n`$ under the weighted rule). In both rules, branch replication preserves exact compatibility with whatever subset of features *is* observed in $`\mathbf{x}`$ along each sub-branch.

### 7.2 Four-Way Ensemble Aggregation

Given the ensemble $`\mathcal{E} = \{(T_b, w_b)\}_{b=1}^{B}`$, let each tree's hard prediction be $`\hat{y}_b(\mathbf{x}) = \arg\max_{c \in \mathcal{C}} p_{T_b}(c \mid \mathbf{x})`$, and let tree weight $`w_b`$ be governed by $`\sigma_{\text{wt}}`$:

$$
w_b = \begin{cases}
  \omega\big(s_{\text{surv}}(T_b)\big), & \text{if } \sigma_{\text{wt}} = \text{True (score-weighted)}, \\
  1, & \text{if } \sigma_{\text{wt}} = \text{False (uniform)},
\end{cases} \qquad W = \sum_{b=1}^{B} w_b,
$$

where the weighting function is set by `tree_weighting`:

$$
\omega(s) = \begin{cases}
  s, & \text{score (default, v1.5.0)}, \\
  \max\!\Big(0,\; \log\dfrac{\bar{s}}{1 - \bar{s}} + \log(K - 1)\Big), \quad \bar{s} = \min\!\big(\max(s, 10^{-3}),\, 1 - 10^{-3}\big), & \text{samme}.
\end{cases}
$$

The `"samme"` weight is the multiclass AdaBoost (SAMME) classifier weight (Zhu et al., 2009): it is zero at chance accuracy $`1/K`$ and grows without bound as $`s \to 1`$. With raw scores, survivors typically have $`s_{\text{surv}} \in [0.8, 1.0]`$, so weights differ by at most $`25\%`$ and score weighting barely changes the vote; the log-odds transform spreads them substantially.

Combining $`\sigma_{\text{soft}} \in \{\text{False}, \text{True}\}`$ and $`\sigma_{\text{wt}} \in \{\text{False}, \text{True}\}`$ defines four ensemble probability estimators for each class $`c \in \mathcal{C}`$:

| Voting Mode | Weighting Mode | Ensemble Class Probability $`P_{\mathcal{E}}(y = c \mid \mathbf{x})`$ |
| :--- | :--- | :--- |
| **Hard** ($`\sigma_{\text{soft}} = \text{False}`$) | **Uniform** ($`\sigma_{\text{wt}} = \text{False}`$) | $`\displaystyle \frac{1}{B} \sum_{b=1}^{B} \mathbb{I}\!\big(\hat{y}_b(\mathbf{x}) = c\big)`$ |
| **Hard** ($`\sigma_{\text{soft}} = \text{False}`$) | **Weighted** ($`\sigma_{\text{wt}} = \text{True}`$) | $`\displaystyle \frac{1}{W} \sum_{b=1}^{B} w_b\,\mathbb{I}\!\big(\hat{y}_b(\mathbf{x}) = c\big)`$ |
| **Soft** ($`\sigma_{\text{soft}} = \text{True}`$) | **Uniform** ($`\sigma_{\text{wt}} = \text{False}`$) | $`\displaystyle \frac{1}{B} \sum_{b=1}^{B} p_{T_b}(c \mid \mathbf{x})`$ |
| **Soft** ($`\sigma_{\text{soft}} = \text{True}`$) | **Weighted** ($`\sigma_{\text{wt}} = \text{True}`$) | $`\displaystyle \frac{1}{W} \sum_{b=1}^{B} w_b\,p_{T_b}(c \mid \mathbf{x})`$ |

**Prior correction (`class_priors`).** Because every tree is trained on class-balanced batches (§3), $`P_{\mathcal{E}}`$ estimates class posteriors under a *uniform* prior $`1/K`$. Given target priors $`\pi = (\pi_c)_{c \in \mathcal{C}}`$ (`class_priors="train"` uses the training class frequencies $`M_c/M`$; a dictionary supplies arbitrary priors; None, the default, applies no correction), the forest output is reweighted by Bayes' rule:

$$
P^{\pi}_ {\mathcal{E}}(y = c \mid \mathbf{x}) = \frac{P_{\mathcal{E}}(y = c \mid \mathbf{x})\,\pi_c}{\sum_{c' \in \mathcal{C}} P_{\mathcal{E}}(y = c' \mid \mathbf{x})\,\pi_{c'}}.
$$

The final predicted class label is the maximum a posteriori class of the (possibly prior-corrected) ensemble distribution:

$$
\hat{y}_ {\mathcal{E}}(\mathbf{x}) = \arg\max_{c \in \mathcal{C}} P^{\pi}_{\mathcal{E}}(y = c \mid \mathbf{x}).
$$

Without correction, the argmax targets balanced accuracy; with $`\pi`$ equal to the deployment frequencies, it targets plain accuracy and yields probabilities under the deployment prior. For binary targets, `tune_decision_threshold` alternatively selects, on held-out labelled data, the threshold $`t^{\ast}`$ on $`P_{\mathcal{E}}(y = c_{+} \mid \mathbf{x})`$ (positive class $`c_{+}`$, by default the minority training class) that maximizes balanced accuracy, preferring thresholds closer to $`0.5`$ among ties; `predict` then labels $`\mathbf{x}`$ positive iff $`P_{\mathcal{E}}(y = c_{+} \mid \mathbf{x}) \ge t^{\ast}`$.

*Note on Leaf Purity and Smoothing:* When trees are grown to full purity ($`d_{\max} = \infty`$, $`s_{\min} = 1`$), queries contain no missing values and $`\alpha = 0`$, each leaf posterior $`p_{\ell}(c)`$ collapses to a one-hot vector, making unweighted soft voting mathematically identical to unweighted hard voting. Even when leaves are impure, they typically hold only $`1`$–$`3`$ training samples, so unsmoothed soft voting is close to hard voting and the ensemble probabilities are coarse and overconfident. Laplace smoothing $`p_{\ell}(c) = (n_{\ell}(c) + \alpha)/(n_{\ell} + K\alpha)`$ (§5.2) with $`\alpha > 0`$ shrinks small leaves toward the uniform distribution in proportion to their size, so that soft voting reflects leaf support (a pure leaf with $`n_{\ell} = 1`$ and $`\alpha = 1`$, $`K = 2`$ contributes $`(2/3, 1/3)`$ rather than $`(1, 0)`$). Setting $`\alpha = 0`$ reproduces v1.5.0. Soft voting also differs from hard voting when missing features trigger multi-leaf replication.

### 7.3 Counterfactual Missing-Value Profiling (`predictMissingValues`)

Beyond passive robustness during prediction, RF-TSMC supports counterfactual evaluation over missing features. Given an incomplete observation $`\mathbf{x}`$ (where at least one feature $`f`$ is missing) and a candidate value dictionary $`\mathcal{V} = \{f : \text{Vals}(f)\}`$ specifying discrete candidate values for features used in the forest ($`\bigcup_{b=1}^B \mathcal{U}(T_b)`$):

1. For every missing feature $`f`$ in $`\mathbf{x}`$ appearing in $`\mathcal{V}`$ and every candidate value $`v \in \text{Vals}(f)`$, construct the counterfactual vector $`\mathbf{x}^{(f \leftarrow v)}`$ obtained by setting $`x_f \leftarrow v`$ while preserving all other entries (including any other simultaneously missing features, which remain handled by branch replication).
2. Evaluate the ensemble posterior profile $`P_{\mathcal{E}}(y = c \mid \mathbf{x}^{(f \leftarrow v)})`$ across all $`c \in \mathcal{C}`$.

This yields two analytical modes:
- **Supervised Imputation (Known Target $`y^*`$):** Select the candidate imputation $`v^* = \arg\max_{v \in \text{Vals}(f)} P_{\mathcal{E}}(y = y^* \mid \mathbf{x}^{(f \leftarrow v)})`$ that maximizes posterior support for the observed ground-truth label.
- **Unsupervised Sensitivity Analysis (Unknown Target):** Inspect whether $`P_{\mathcal{E}}(\cdot \mid \mathbf{x}^{(f \leftarrow v)})`$ varies across $`v \in \text{Vals}(f)`$. If the profile is invariant across all candidates, the missing attribute lies outside the active decision paths conditioned on the observed attributes of $`\mathbf{x}`$ and cannot influence classification confidence.

---

## 8. Structural Interpretability and Compositional Forest Operations

### 8.1 Survival-Based Feature Importance and Co-occurrence

For each surviving tree $`T_b \in \mathcal{E}`$, let $`\mathcal{U}(T_b) \subseteq F_b`$ denote the set of **actually utilized features**—i.e., the distinct features appearing in at least one internal split node of $`T_b`$ (a feature present in the sampled queue $`F_b`$ that was never split upon before reaching leaf nodes is excluded from $`\mathcal{U}(T_b)`$).

Because trees only enter $`\mathcal{E}`$ after surviving Monte Carlo selection on $`\mathcal{I}_V`$, the presence of $`f \in \mathcal{U}(T_b)`$ is itself an empirical indicator of discriminative utility. RF-TSMC defines four structural interpretability metrics over any forest $`\mathcal{E}`$ of size $`B = |\mathcal{E}|`$:

**1. Marginal Survival Importance (`featImportance`).**

$$
I_ {\mathcal{E}}(f) = \frac{1}{B} \sum_{b=1}^{B} \mathbb{I}\!\big(f \in \mathcal{U}(T_b)\big) \in [0, 1].
$$

Unlike Gini-based Mean Decrease in Impurity (MDI), which is biased toward high-cardinality continuous features with many split points, $`I_{\mathcal{E}}(f)`$ treats numeric and categorical attributes symmetrically as binary survival counts across validation-filtered trees.

**2. Conditional Mean Survival Score (`featScoreMean`).**

$$
\bar{S}_ {\mathcal{E}}(f) = \frac{\sum_{b=1}^{B} s_ {\text{surv}}(T_b)\,\mathbb{I}\!\big(f \in \mathcal{U}(T_b)\big)}{\sum_{b=1}^{B} \mathbb{I}\!\big(f \in \mathcal{U}(T_b)\big)},
$$

measuring the average generalization score of surviving trees that rely on feature $`f`$ (zero if $`f`$ never appears in $`\mathcal{E}`$).

**3. Pairwise Feature Co-occurrence Importance (`featPairImportance`).** For every unordered pair $`\{f_i, f_j\} \subseteq \mathcal{F}`$ with $`i < j`$:

$$
I_ {\mathcal{E}}(f_i, f_j) = \frac{1}{B} \sum_{b=1}^{B} \mathbb{I}\!\big(f_i \in \mathcal{U}(T_b) \land f_j \in \mathcal{U}(T_b)\big) \in [0, 1].
$$

**4. Structural Co-occurrence Matrix (`featCorrDataFrame`).** A symmetric $`d \times d`$ matrix $`\mathbf{M}_{\mathcal{E}} \in [0, 1]^{d \times d}`$ defined by:

$$
\big[\mathbf{M}_ {\mathcal{E}}\big]_ {ij} = \begin{cases}
  I_ {\mathcal{E}}(f_i), & i = j, \\
  I_ {\mathcal{E}}(f_i, f_j), & i \neq j.
\end{cases}
$$

Notice that $`\mathbf{M}_{\mathcal{E}}`$ is positive semidefinite because it can be written as the normalized Gram matrix $`\frac{1}{B}\mathbf{U}^\top \mathbf{U}`$ of the binary tree-feature incidence matrix $`\mathbf{U} \in \{0, 1\}^{B \times d}`$ with entries $`U_{bi} = \mathbb{I}(f_i \in \mathcal{U}(T_b))`$.

*Limitations of survival-frequency measures.* $`I_{\mathcal{E}}(f)`$ is computed from only $`B`$ survivors, so its resolution is $`1/B`$ ($`1/16`$ with the defaults). It is also **positionally confounded**: by §5.4, $`f \in \mathcal{U}(T_b)`$ essentially means that $`f`$ occupied one of the first $`\approx L_{\text{eff}}`$ positions of $`F_b`$, which happens with probability $`\approx \min(1, L_{\text{eff}}/d)`$ for every feature regardless of its utility, and survival filtering on a noisy selection score (§6.1) only modestly tilts these frequencies. The co-occurrence measures inherit both issues.

#### Candidate-Based Importance (v1.6.0)
Every Monte Carlo candidate, accepted or not, is logged as a record $`(F_j, \mathcal{U}(T_j), s_{\text{sel},j}, s_{\text{sc},j}, b_j)`$ with its slot index $`b_j`$. Let $`s_j`$ denote its score ($`s_{\text{sc},j}`$ if available, else $`s_{\text{sel},j}`$), and let $`L \approx L_{\text{eff}}`$ be a prefix length. Since $`F_j`$ is drawn uniformly at random (§4), the indicator

$$
Z_j(f) = \mathbb{I}\big(f \in \{F_j[0], \dots, F_j[L-1]\}\big)
$$

is a **randomized treatment** assigned independently of the data, and the difference in mean candidate score

$$
\Delta(f) = \frac{\sum_j s_j\, Z_j(f)}{\sum_j Z_j(f)} - \frac{\sum_j s_j\,(1 - Z_j(f))}{\sum_j (1 - Z_j(f))}
$$

is an unbiased estimate of the average effect on the candidate score of placing $`f`$ near the root (`featImportanceMC`). Confidence intervals are obtained by a bootstrap that resamples slots (clusters) rather than candidates, because candidates of the same slot share a batch. Using $`f \in \mathcal{U}(T_j)`$ as the treatment instead would be slightly confounded, since whether $`f`$ is actually used depends on the data through the purity stopping rule. Main and pairwise effects are estimated jointly by a ridge regression (`featPairImportanceMC`)

$$
s_j \approx \beta_0 + \sum_{f} \beta_f\, Z_j(f) + \sum_{f < g} \beta_{fg}\, Z_j(f)\, Z_j(g),
$$

whose interaction coefficients $`\beta_{fg}`$ replace the co-occurrence measure `featPairImportance` for the purpose of detecting synergistic pairs (pair terms are restricted to the features with the largest main effects). Because the candidate log contains $`\bar{R}\,B`$ records rather than $`B`$, these estimates have far finer resolution than $`I_{\mathcal{E}}(f)`$.

### 8.2 Local Sample- and Class-Conditional Explainability

To explain *why* the forest supports a specific class $`c \in \mathcal{C}`$ for an individual sample $`\mathbf{x}`$, RF-TSMC defines the **class-consistent sub-forest**:

$$
\mathcal{E}(\mathbf{x}, c) = \left\{ T_b \in \mathcal{E} : \hat{y}_ {b}(\mathbf{x}) = c \right\}.
$$

Evaluating $`I_{\mathcal{E}(\mathbf{x}, c)}(f)`$, $`\bar{S}_{\mathcal{E}(\mathbf{x}, c)}(f)`$, $`I_{\mathcal{E}(\mathbf{x}, c)}(f_i, f_j)`$, $`\mathbf{M}_{\mathcal{E}(\mathbf{x}, c)}`$, and feature-count statistics $`\{|\mathcal{U}(T_b)| : T_b \in \mathcal{E}(\mathbf{x}, c)\}`$ yields **local, sample-specific structural explanations** showing which features and synergistic feature pairs formed the decision paths of trees predicting class $`c`$ for $`\mathbf{x}`$.

### 8.3 Forest Merging and Canonical Deduplication

Because each tree $`T_b`$ is a self-contained classifier equipped with its validation score $`s_{\text{surv}}(T_b)`$, separately trained models $`\mathcal{E}_1`$ and $`\mathcal{E}_2`$ sharing identical feature space $`\mathcal{F}`$ and class space $`\mathcal{C}`$ can be composed natively via `mergeForest`:

- **Additive Union (`by="add"`):** $`\mathcal{E}_{\text{new}} = \mathcal{E}_1 \cup \mathcal{E}_2`$, enabling incremental model updates (e.g., combining historical trees trained on past data regimes with newly fitted trees on recent batches to mitigate concept drift and recency bias).
- **Elitist Score Selection (`by="score"`):** Sort $`\mathcal{E}_1 \cup \mathcal{E}_2`$ in descending order of $`s_{\text{surv}}(T)`$ and retain the top $`N`$ trees.
- **Random Subsampling (`by="random"`):** Sample $`N`$ trees uniformly without replacement from $`\mathcal{E}_1 \cup \mathcal{E}_2`$.

To preserve structural diversity after repeated merging or small feature-space training, `drop_duplicated_trees` hashes each tree's canonical key-sorted dictionary representation using MD5 ($`\text{MD5}(\text{JSON}_{\text{sorted}}(T))`$) and removes exact structural duplicates while preserving survival scores in lockstep.

*Serialization formats.* Every tree carries a `tree_format` tag: $`1`$ for trees produced by v1.5.0 and older, $`2`$ for v1.6.0 trees (midpoint thresholds, stored counts $`n_{\ge}, n_{<}, n_{\ell}`$, optional `categorical_set` nodes). Because thresholds, and therefore structural hashes, differ between formats for the same data, `mergeForest` and `drop_duplicated_trees` refuse forests with mixed formats unless `allow_mixed_formats=True` is passed. Format-1 trees remain usable for inference (the weighted missing-value rule falls back to $`1/2`$–$`1/2`$ branch weights). In both engines, categorical split values (single categories or category sets) are serialized as the category values themselves.

---

## 9. Asymptotic Complexity and Comparative Theoretical Analysis

### 9.1 Computational Complexity

Let $`n_T = |\mathcal{I}_T| = K \cdot n'_{\text{train}}`$ and $`n_V = |\mathcal{I}_V| = K \cdot n'_{\text{pool}}`$ denote the training and validation (selection plus scoring) batch sizes per tree, $`L_T \le d_{\max}`$ the realized tree depth, and $`\bar{R} \ge 1`$ the expected number of Monte Carlo candidate evaluations per surviving tree.

**Single-Tree Induction (`PlantTree`).** For numeric attributes, the distinct-value midpoint rule (§5.1) sorts the node values, taking $`O(m \log m)`$ at a node of size $`m \le n_T`$; the legacy median rule takes $`O(m)`$ expected time via quickselect (`std::nth_element` in C++). For categorical attributes, modal frequency counting takes $`O(m)`$ (ordered set splits add an $`O(m \log m)`$ sort). Label-aware splits (§5.5) evaluate at most three cuts per feature at $`O(m)`$ each, and level $`2`$ multiplies the per-node work by at most $`H`$ (more only when the first $`H`$ features yield no valid split). If at most $`c_{\text{rot}} \le k`$ queue rotations occur per node and splits roughly bisect $`m`$, total work across all levels of a single tree is:

$$
O\!\big(c_ {\text{rot}} \, n_T \log n_T\big) \quad \text{(linear-median selection, legacy)}, \qquad O\!\big(c_ {\text{rot}} \, n_T \log^2 n_T\big) \quad \text{(sort-based midpoint, default)}.
$$

Crucially, **this cost does not depend on the full dataset size $`M`$**.

**Single-Tree Validation (`Validate`).** Evaluating $`n_V`$ balanced validation samples down depth $`L_T`$ requires $`O(n_V \, L_T)`$ operations when features are observed.

**Full Forest Training (`fit`).** Across $`B`$ independent trees, total sequential work is:

$$
O\!\Big(B \cdot \bar{R} \cdot \big(c_ {\text{rot}} \, n_T \log n_T + n_V \, L_T\big)\Big).
$$

Because all $`B`$ tree-selection loops are mutually independent, training scales with near-linear speedup across $`P`$ parallel workers ($`O(B/P)`$ parallel steps).

**Inference (`predict`).** On complete rows, inference complexity is $`O(B \, L_T)`$ per sample. When a query encounters $`h`$ missing split features along its active paths, replication visits at most $`2^h`$ leaves in a tree (bounded by total leaves $`\le n_T`$), giving worst-case $`O(B \cdot \min(2^h L_T, n_T))`$ per sample under either missing-value strategy (the weighted rule only multiplies path weights).

**Ensemble Selection (§6.4).** With a library of $`N_{\text{lib}} = \bar{R} B`$ candidates and $`n_H = K\, n'_{\text{ens}}`$ holdout rows, scoring the library costs $`O(N_{\text{lib}}\, n_H\, L_T)`$, and each greedy step evaluates all candidates at $`O(N_{\text{lib}}\, n_H\, K)`$, for $`O(B\, N_{\text{lib}}\, n_H\, K)`$ in total.

### 9.2 Theoretical Comparison with Related Tree Ensembles

| Property | Classical Random Forest (Breiman, 2001) | Extra-Trees (Geurts et al., 2006) | RF-TSMC (`random-forest-mc`) |
| :--- | :--- | :--- | :--- |
| **Sample Strategy** | Full-size bootstrap ($`M`$ with replacement) | Full dataset ($`M`$, usually without bootstrap) | Class-balanced disjoint micro-batches ($`\mathcal{I}_T, \mathcal{I}_V`$) |
| **Class-Imbalance Sensitivity** | Mirrors empirical skew in $`\mathcal{D}`$ | Mirrors empirical skew in $`\mathcal{D}`$ | Training invariant ($`1/K`$ prior in $`\mathcal{I}_T, \mathcal{I}_V`$); outputs need prior correction (§7.2) |
| **Split Criterion** | Greedy Gini / Entropy maximization | Impurity evaluation over random thresholds | Impurity-free distinct-value midpoint at the count-median (num.) / mode (cat.); optional Gini among percentile cuts (§5.5) |
| **Feature Randomization** | Fixed $`m_{\text{try}}`$ resampled at every node | Fixed $`K_{\text{attr}}`$ resampled at every node | Random cardinality $`k \in [f_{\min}, f_{\max}]`$ & cyclic queue $`F`$ per tree (one feature per level; effective prefix length $`\approx L_{\text{eff}}`$) |
| **Structural Optimization** | Greedy local node selection | Greedy selection among random cut-points | Monte Carlo tree-level acceptance–rejection on $`\mathcal{I}_{\text{sel}}`$, or greedy ensemble selection (§6.4) |
| **Tree Acceptance Filter** | None (all induced trees kept) | None (all induced trees kept) | Survival threshold $`\tau \ge \tau_{\min} = 1/K + \varepsilon`$ (best-of-$`R_{\max}`$ or $`\Delta\tau`$ decay), attempt cap $`A_{\max}`$ |
| **Missing-Data Inference** | Requires imputation / surrogate splits | Requires imputation / surrogate splits | Native dual-branch replication with training-fraction weights (C4.5-style) |
| **Feature Importance** | MDI / Permutation importance | MDI / Permutation importance | Survival frequency $`I_{\mathcal{E}}(f)`$, co-occurrence $`\mathbf{M}_{\mathcal{E}}`$; randomized prefix effects $`\Delta(f)`$ & ridge pair effects over all candidates |
| **Model Composition** | Tree union (no per-tree validation score) | Tree union (no per-tree validation score) | Native merging (`add`, `score`, `random`) & MD5 deduplication |

RF-TSMC combines several ideas that appear separately in prior work. The following table lists the closest methods and how RF-TSMC relates to each.

| Related Method | Shared Idea | Difference in RF-TSMC |
| :--- | :--- | :--- |
| **Balanced Random Forest** (Chen, Liaw & Breiman, 2004) | Each tree is grown on a class-balanced undersample (minority-size draw per class); the closest prior method for imbalance handling | Balanced batches are much smaller ($`n_{\text{train}}`$ per class), are paired with a balanced out-of-batch selection set, and trees are filtered by validation; splits are median/mode rather than CART |
| **Median forests** (Duroux & Scornet, 2018) | Splits at the empirical median of the node; consistency analysis of such forests | RF-TSMC uses distinct-value midpoints at the count-median along a fixed feature queue; consistency requires the subsample size to grow with $`M`$, which the default fixed $`n_{\text{train}}`$ does not satisfy (hence `n_train_frac`, §3) |
| **Ensemble selection** (Caruana et al., 2004) | Greedy forward selection with replacement from a library of models, optimizing the ensemble metric on a holdout | Used as the optional `selection="ensemble"` assembly (§6.4), with the library formed by all Monte Carlo candidates of all slots |
| **Oblivious decision trees** (e.g., CatBoost; Prokhorenkova et al., 2018) | One split feature per tree level | In RF-TSMC obliviousness in feature choice arises implicitly from the shared rotated queue (§5.4); thresholds remain node-specific, and levels diverge where a split fails |
| **C4.5** (Quinlan, 1993) | Fractional propagation of cases with missing values, weighted by training branch frequencies | Adopted as the default `missing_strategy="weighted"` at inference (§7.1); training rows with missing values are still excluded |
| **SAMME** (Zhu et al., 2009) | Classifier weight $`\log\frac{s}{1-s} + \log(K-1)`$ for multiclass boosting | Used as the optional `tree_weighting="samme"` vote weight from the survival score (§7.2), without sequential reweighting of samples |

---

## 10. Theoretical Assumptions, Trade-Offs, and Design Considerations

A rigorous understanding of RF-TSMC also illuminates the theoretical conditions governing its behavior:

1. **Resolution Quantization and Noise of Validation Scores:**  
   Since $`|\mathcal{I}_{\text{sel}}| = K \cdot n'_{\text{sel}}`$, the selection score is quantized to integer multiples of $`(K \cdot n'_{\text{sel}})^{-1}`$ and has binomial standard deviation $`\approx 0.10`$ at the defaults. When $`K \cdot n'_{\text{sel}}`$ is small, many candidates tie, best-of-$`R_{\max}`$ selection is driven largely by noise, and a survival score measured on the selection set is optimistically biased (§6.1). Increasing `val_pool_pclass` refines the resolution and reduces the bias of the Monte Carlo objective at the cost of $`O(n_{\text{pool}} L_T)`$ validation time per trial; `score_split` $`> 0`$ removes the bias from the reported score at the cost of fewer selection rows (subject to $`n'_{\text{train}} + n'_{\text{pool}} \le M_{\min}`$).
2. **Bias–Variance Trade-Off in Batch Sizes ($`n_{\text{train}}`$ vs. $`n_{\text{pool}}`$):**  
   Larger $`n_{\text{train}}`$ enables deeper decision boundaries and finer local partitioning within each tree, reducing individual-tree bias (and, through `n_train_frac`, restores the growing-subsample condition required for consistency of median forests); however, for small datasets where $`M_{\min}`$ is constrained, increasing $`n_{\text{train}}`$ shrinks the validation pool (or forces overlapping training pools across trees), increasing the variance of the selection filter.
3. **Axis-Aligned Univariate Median/Mode Inductive Bias:**  
   Because internal splits test single features at their count-median midpoint or mode without greedy multi-threshold scanning (unless label-aware splits are enabled), linear oblique boundaries or sharply asymmetric bimodal thresholds on a single feature may require multiple cyclic passes through $`F`$ (i.e., deeper paths or multiple splits along the same feature at different depths) to approximate. With one feature per level and $`L_{\text{eff}} \approx 5`$–$`6`$, each tree uses only about $`L_{\text{eff}}`$ distinct features, whatever $`k`$ is.
4. **Minimum Class Cardinality Requirement:**  
   Constructing disjoint non-empty $`\mathcal{I}_T`$ and $`\mathcal{I}_{\text{sel}}`$ requires $`M_{\min} \ge 2`$ (and practically $`M_{\min} \ge n_{\text{train}} + n_{\text{pool}}`$, plus $`n'_{\text{ens}}`$ under ensemble selection, to realize the full configured batch sizes). On tiny or nearly separable-by-chance problems, the threshold floor may be unreachable on most batches; the attempt cap then bounds training time and the affected slots are flagged.
5. **Combinatorial Complexity of Dual-Branch Replication:**  
   When an inference query omits or masks $`h`$ features along a tree's paths, leaf collection grows as $`O(2^h)`$ up to the tree's leaf count ($`O(n_T)`$). Because $`n_T`$ is deliberately bounded by balanced micro-batching, worst-case traversal remains bounded by tree size rather than full dataset size $`M`$. The weighted rule assumes the missingness mechanism is ignorable at the split (missing values distributed like the training rows at that node); informative missingness is not modeled.
6. **Balanced Training versus Calibrated Outputs:**  
   Balanced batches make split selection independent of class skew but produce posteriors for a uniform prior. Unless `class_priors` or a tuned decision threshold is used, the forest optimizes balanced accuracy rather than plain accuracy, and its probabilities are not calibrated to the deployment class frequencies.

---

## 11. Summary

RF-TSMC formulates ensemble classification as a two-level stochastic architecture: an **inner deterministic constructor** that builds compact, class-balanced, impurity-free decision trees along cyclic feature queues, using distinct-value midpoint splits that inference reproduces exactly and that, in practice, consume one queue feature per level, paired with an **outer Monte Carlo selection mechanism** that filters feature-sequence prefixes by out-of-sample balanced accuracy above a chance-level floor, or alternatively assembles the forest by greedy ensemble selection over all candidates. This separation yields class-balanced training that is insensitive to severe imbalance (with prior correction available for the outputs), dataset-size-independent tree induction, embarrassingly parallel training, imputation-free missing-value inference via training-weighted path replication, survival-driven and randomized candidate-based explainability, and native model compositionality. The v1.6.0 revisions make explicit the limits of the method on small validation sets, where selection is noise-dominated, and provide opt-in mechanisms (larger and split validation pools, label-aware splits, smoothing, prior correction and ensemble selection) to address them.

---

## References

1. Breiman, L. (2001). *Random Forests*. Machine Learning, Springer.
2. Geurts, P., Ernst, D., & Wehenkel, L. (2006). *Extremely randomized trees*. Machine Learning, Springer.
3. Ho, T. K. (1998). *The random subspace method for constructing decision forests*. IEEE Transactions on Pattern Analysis and Machine Intelligence.
4. Oliveira, I. *random-forest-mc: Random Forest with Tree Selection Monte Carlo Based (v1.6.0)*. GitHub repository: [https://github.com/ysraell/random-forest-mc](https://github.com/ysraell/random-forest-mc).
5. Oliveira, I. *Laboratory of Decision Tree and Random Forest (`random-forest-lab`)*. GitHub repository: [https://github.com/ysraell/random-forest-lab](https://github.com/ysraell/random-forest-lab).
6. Chen, C., Liaw, A., & Breiman, L. (2004). *Using Random Forest to Learn Imbalanced Data*. Technical Report 666, Department of Statistics, University of California, Berkeley.
7. Duroux, R., & Scornet, E. (2018). *Impact of subsampling and tree depth on random forests*. ESAIM: Probability and Statistics, 22, 96–128.
8. Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A. (2004). *Ensemble selection from libraries of models*. Proceedings of the 21st International Conference on Machine Learning (ICML).
9. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). *CatBoost: unbiased boosting with categorical features*. Advances in Neural Information Processing Systems (NeurIPS) 31.
10. Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.
11. Zhu, J., Zou, H., Rosset, S., & Hastie, T. (2009). *Multi-class AdaBoost*. Statistics and Its Interface, 2(3), 349–360.
