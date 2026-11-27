"""
Ensemble selection from a library of candidate trees (Caruana et al., 2004).

Shared by the Python and C++ engines: it only needs the candidates' class probabilities
on a common holdout set.
"""

from typing import List

import numpy as np


def _metric(probs: np.ndarray, y: np.ndarray, metric: str) -> float:
    """Higher is better. `probs` is (n_rows, K)."""
    if metric == "logloss":
        p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
        return float(np.mean(np.log(p)))
    pred = np.argmax(probs, axis=1)
    recalls = [np.mean(pred[y == c] == c) for c in np.unique(y)]
    return float(np.mean(recalls))


def ensemble_score(tree_probs: np.ndarray, y: np.ndarray, metric: str = "balanced_accuracy") -> float:
    """Score of the soft-voting ensemble of `tree_probs` (n_trees, n_rows, K)."""
    return _metric(tree_probs.mean(axis=0), y, metric)


def ensemble_select(
    cand_probs: np.ndarray,
    y: np.ndarray,
    n_select: int,
    n_init: int = 3,
    metric: str = "balanced_accuracy",
) -> List[int]:
    """Greedy forward selection with replacement.

    Starts from the `n_init` best individual candidates and repeatedly adds the candidate that most
    improves the soft-voting ensemble on the holdout, until `n_select` trees are chosen or no candidate
    keeps the score from dropping.

    Args:
        cand_probs: (n_candidates, n_rows, K) class probabilities on the holdout.
        y: (n_rows,) integer class codes.
        n_select: Maximum forest size.
        n_init: Number of best individual candidates to start from.
        metric: 'balanced_accuracy' or 'logloss'.

    Returns:
        Indices of the chosen candidates (a candidate may appear more than once).
    """
    n_cand = cand_probs.shape[0]
    if n_cand == 0:
        return []
    cand_probs = cand_probs.astype(np.float64)

    def key(probs):
        # Plateaus of a discrete metric are broken by the mean log-probability of the true class.
        return _metric(probs, y, metric), _metric(probs, y, "logloss")

    individual = [key(cand_probs[i]) for i in range(n_cand)]
    order = sorted(range(n_cand), key=lambda i: individual[i], reverse=True)
    chosen = [int(i) for i in order[: max(1, min(n_init, n_select, n_cand))]]
    total = cand_probs[chosen].sum(axis=0)
    current = key(total / len(chosen))
    while len(chosen) < n_select:
        m = len(chosen) + 1
        keys = [key((total + cand_probs[i]) / m) for i in range(n_cand)]
        best = max(range(n_cand), key=lambda i: keys[i])
        if keys[best][0] + 1e-12 < current[0]:
            break
        chosen.append(best)
        total += cand_probs[best]
        current = keys[best]
    return chosen


def mean_pairwise_agreement(tree_probs: np.ndarray) -> float:
    """Average fraction of rows on which two trees predict the same class."""
    preds = np.argmax(tree_probs, axis=2)
    n = preds.shape[0]
    if n < 2:
        return 1.0
    agree = [np.mean(preds[i] == preds[j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(agree))


# EOF
