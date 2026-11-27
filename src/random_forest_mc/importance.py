"""
Feature importance from every Monte Carlo candidate tree.

The candidate loop draws feature sequences F uniformly at random, so it is a randomized
experiment: "feature f is among the first L positions of F" is a treatment assigned at random,
and the difference in mean candidate score estimates the effect of making f available near the root.
"""

from itertools import combinations
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _scores(candidate_log: List[dict]) -> np.ndarray:
    return np.array(
        [rec["s_score"] if rec.get("s_score") is not None else rec["s_sel"] for rec in candidate_log],
        dtype=float,
    )


def _prefix_matrix(candidate_log: List[dict], features: List[str], prefix_len: int) -> np.ndarray:
    col = {f: j for j, f in enumerate(features)}
    X = np.zeros((len(candidate_log), len(features)))
    for i, rec in enumerate(candidate_log):
        for f in rec["F"][:prefix_len]:
            if f in col:
                X[i, col[f]] = 1.0
    return X


def _delta(X: np.ndarray, s: np.ndarray) -> np.ndarray:
    n_in = X.sum(axis=0)
    n_out = len(s) - n_in
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_in = (X * s[:, None]).sum(axis=0) / n_in
        mean_out = ((1 - X) * s[:, None]).sum(axis=0) / n_out
    return mean_in - mean_out


def randomized_importance(
    candidate_log: List[dict],
    features: List[str],
    prefix_len: int,
    n_boot: int = 200,
    seed: Optional[int] = 0,
) -> pd.DataFrame:
    """Δ(f) = mean score | f in prefix  −  mean score | f not in prefix, with a slot-cluster bootstrap CI.

    Args:
        candidate_log: Records with keys 'F', 's_sel', 's_score' and 'slot'.
        features: All feature names.
        prefix_len: Number of leading positions of F that define the treatment (about L_eff).
        n_boot: Bootstrap resamples (0 disables the CI).
        seed: Seed of the bootstrap.

    Returns:
        DataFrame indexed by feature with columns delta, ci_low, ci_high, n_in, sorted by delta.
    """
    if not candidate_log:
        raise ValueError("Empty candidate log: fit the model first (the log is not saved by default).")
    s = _scores(candidate_log)
    X = _prefix_matrix(candidate_log, features, prefix_len)
    delta = _delta(X, s)
    out = pd.DataFrame({"delta": delta, "n_in": X.sum(axis=0).astype(int)}, index=features)
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        slots = np.array([rec.get("slot", i) for i, rec in enumerate(candidate_log)])
        groups = [np.nonzero(slots == g)[0] for g in np.unique(slots)]
        boots = np.empty((n_boot, len(features)))
        for b in range(n_boot):
            pick = rng.integers(0, len(groups), len(groups))
            rows = np.concatenate([groups[g] for g in pick])
            boots[b] = _delta(X[rows], s[rows])
        out["ci_low"] = np.nanpercentile(boots, 2.5, axis=0)
        out["ci_high"] = np.nanpercentile(boots, 97.5, axis=0)
    return out.sort_values("delta", ascending=False)


def ridge_importance(
    candidate_log: List[dict],
    features: List[str],
    prefix_len: int,
    alpha: float = 1.0,
    pairwise: bool = True,
    max_pair_features: int = 30,
) -> Dict[str, Any]:
    """Ridge regression of the candidate score on feature-in-prefix indicators and pairwise products.

    Pairwise terms are limited to the `max_pair_features` features with the largest main effect.

    Returns:
        {'intercept': float, 'main': {feature: coef}, 'pairs': {(f, g): coef}}
    """
    if not candidate_log:
        raise ValueError("Empty candidate log: fit the model first (the log is not saved by default).")
    s = _scores(candidate_log)
    X = _prefix_matrix(candidate_log, features, prefix_len)
    names: List[Any] = list(features)
    if pairwise and len(features) > 1:
        main = _delta(X, s)
        top = [int(j) for j in np.argsort(-np.nan_to_num(main, nan=-np.inf))[:max_pair_features]]
        top.sort()
        pair_cols = []
        for a, b in combinations(top, 2):
            pair_cols.append(X[:, a] * X[:, b])
            names.append((features[a], features[b]))
        if pair_cols:
            X = np.hstack([X, np.column_stack(pair_cols)])
    x_mean = X.mean(axis=0)
    y_mean = s.mean()
    Xc = X - x_mean
    yc = s - y_mean
    n, p = Xc.shape
    if p <= n:
        beta = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(p), Xc.T @ yc)
    else:
        beta = Xc.T @ np.linalg.solve(Xc @ Xc.T + alpha * np.eye(n), yc)
    intercept = float(y_mean - x_mean @ beta)
    main_coefs = {name: float(b) for name, b in zip(names, beta) if not isinstance(name, tuple)}
    pair_coefs = {name: float(b) for name, b in zip(names, beta) if isinstance(name, tuple)}
    return {"intercept": intercept, "main": main_coefs, "pairs": pair_coefs}


# EOF
