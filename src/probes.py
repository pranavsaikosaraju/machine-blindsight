"""Linear probes on cached activations.

For each task and layer we fit a logistic-regression probe that predicts the
ground-truth label from the final-token residual stream. Evaluation is
leave-one-template-out: the probe is always tested on a prompt wording it never
saw during training, so it cannot rely on memorising one phrasing.

Controls implemented here:
* shuffled labels  (a "control task": a probe that scores well here is too powerful)
* layer-0 baseline (token embeddings only: surface features of the prompt)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _fit_predict(X_tr, y_tr, X_te, C=0.05, seed=0):
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    clf.fit(sc.transform(X_tr), y_tr)
    return clf.decision_function(sc.transform(X_te))


def oof_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float = 0.05,
               seed: int = 0) -> np.ndarray:
    """Out-of-fold decision scores, holding out one group (template) at a time."""
    scores = np.zeros(len(y), dtype=np.float32)
    for g in np.unique(groups):
        te = groups == g
        scores[te] = _fit_predict(X[~te], y[~te], X[te], C=C, seed=seed)
    return scores


def probe_all_layers(acts: np.ndarray, df: pd.DataFrame, C: float = 0.05, seed: int = 0,
                     shuffled_control: bool = True) -> tuple[pd.DataFrame, dict]:
    """Run probes for every task x layer.

    Returns
    -------
    layer_df : one row per (task, layer) with accuracy on all / plain / distractor items,
               and the shuffled-label control accuracy.
    oof      : {task: array [n_task_items, n_layers] of out-of-fold probe scores}
    """
    rows, oof = [], {}
    rng = np.random.default_rng(seed)
    for task, sub in df.groupby("task", sort=False):
        idx = sub.index.to_numpy()
        y = sub.label.to_numpy()
        groups = sub.template.to_numpy()
        cond = sub.condition.to_numpy()
        n_layers = acts.shape[1]
        task_scores = np.zeros((len(idx), n_layers), dtype=np.float32)
        y_shuf = rng.permutation(y)
        for L in range(n_layers):
            X = acts[idx, L, :].astype(np.float32)
            s = oof_scores(X, y, groups, C=C, seed=seed)
            task_scores[:, L] = s
            pred = (s > 0).astype(int)
            row = {
                "task": task, "layer": L,
                "probe_acc": (pred == y).mean(),
                "probe_acc_plain": (pred == y)[cond == "plain"].mean(),
                "probe_acc_distractor": (pred == y)[cond == "distractor"].mean(),
            }
            if shuffled_control:
                s_ctrl = oof_scores(X, y_shuf, groups, C=C, seed=seed)
                row["control_shuffled_acc"] = ((s_ctrl > 0).astype(int) == y_shuf).mean()
            rows.append(row)
        oof[task] = task_scores
    return pd.DataFrame(rows), oof


def mass_mean_direction(acts_layer: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Difference of class means (true minus false). Often more causally useful than
    logistic-regression weights for steering (Marks & Tegmark, 2023)."""
    a = acts_layer.astype(np.float32)
    d = a[y == 1].mean(0) - a[y == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-8)
