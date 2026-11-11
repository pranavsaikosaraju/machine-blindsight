"""Metrics, statistics and figures for Project 1.

Key quantities (per task, and per condition):

report_acc       share of items the model answers correctly
probe_acc        held-out accuracy of the probe at the chosen layer
report_gap       probe_acc - report_acc                                (H1)
blindsight_rate  P(probe right | report wrong), bootstrap 95% CI       (H2)
conf_auroc       how well report confidence separates ordinary-correct
                 items from blindsight items (0.5 = not at all)        (H3)

Layer choice: the layer with the best held-out probe accuracy on PLAIN items.
Blindsight is then measured on all items. Because the probe scores are
out-of-fold across templates, choosing among layers adds only mild optimism;
the README reports this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Reference categorical palette (fixed order, validated for CVD separation)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3df"


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def choose_layers(layer_df: pd.DataFrame) -> dict:
    best = layer_df.loc[layer_df.groupby("task").probe_acc_plain.idxmax()]
    return dict(zip(best.task, best.layer))


def attach_probe(df: pd.DataFrame, oof: dict, best_layers: dict) -> pd.DataFrame:
    df = df.copy()
    df["probe_score"] = np.nan
    df["probe_layer"] = -1
    for task, sub in df.groupby("task", sort=False):
        L = best_layers[task]
        df.loc[sub.index, "probe_score"] = oof[task][:, L]
        df.loc[sub.index, "probe_layer"] = L
    df["probe_pred"] = (df.probe_score > 0).astype(int)
    df["probe_correct"] = (df.probe_pred == df.label).astype(int)
    df["report_correct"] = (df.report == df.label).astype(int)
    df["report_conf"] = np.where(df.report == 1, df.p_pos, 1 - df.p_pos)
    df["blindsight"] = ((df.report_correct == 0) & (df.probe_correct == 1)).astype(int)
    return df


def summarize(df: pd.DataFrame, layer_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, cond), sub in list(df.groupby(["task", "condition"], sort=False)) + \
            [((t, "all"), s) for t, s in df.groupby("task", sort=False)]:
        L = int(sub.probe_layer.iloc[0])
        wrong = sub[sub.report_correct == 0]
        right = sub[sub.report_correct == 1]
        bs = wrong.probe_correct.to_numpy()
        lo, hi = bootstrap_ci(bs)
        bs_items = sub[sub.blindsight == 1]
        try:
            # positive class = ordinary correct; score = confidence
            y = np.r_[np.ones(len(right)), np.zeros(len(bs_items))]
            s = np.r_[right.report_conf, bs_items.report_conf]
            auroc = roc_auc_score(y, s) if len(bs_items) >= 5 and len(right) >= 5 else np.nan
        except ValueError:
            auroc = np.nan
        lrow = layer_df[(layer_df.task == task) & (layer_df.layer == L)].iloc[0]
        l0 = layer_df[(layer_df.task == task) & (layer_df.layer == 0)].iloc[0]
        rows.append({
            "task": task, "condition": cond, "n": len(sub), "layer": L,
            "report_acc": sub.report_correct.mean(),
            "probe_acc": sub.probe_correct.mean(),
            "report_gap": sub.probe_correct.mean() - sub.report_correct.mean(),
            "n_report_wrong": len(wrong),
            "blindsight_rate": bs.mean() if len(bs) else np.nan,
            "blindsight_ci_lo": lo, "blindsight_ci_hi": hi,
            "probe_acc_given_report_right": right.probe_correct.mean() if len(right) else np.nan,
            "conf_ordinary_correct": right.report_conf.mean() if len(right) else np.nan,
            "conf_blindsight": bs_items.report_conf.mean() if len(bs_items) else np.nan,
            "conf_auroc": auroc,
            "layer0_probe_acc": l0.probe_acc,
            "control_shuffled_acc": lrow.get("control_shuffled_acc", np.nan),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures (matplotlib, static PNGs for the repo)
# ---------------------------------------------------------------------------

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_layers(layer_df, summary, out, model_name):
    import matplotlib.pyplot as plt
    tasks = list(dict.fromkeys(layer_df.task))
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.4 * len(tasks), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, task in zip(axes, tasks):
        d = layer_df[layer_df.task == task]
        conds = (("plain", SERIES[0], "no hint"), ("misleading", SERIES[1], "misleading hint"),
                 ("consistent", SERIES[2], "consistent hint"))
        for cond, c, name in conds:
            col = f"probe_acc_{cond}"
            if col not in d:
                continue
            ax.plot(d.layer, d[col], color=c, lw=2, label=f"Probe, {name}")
            r = summary[(summary.task == task) & (summary.condition == cond)].report_acc
            if len(r):
                ax.axhline(r.iloc[0], color=c, lw=1.2, ls="--")
        if "control_shuffled_acc" in d:
            ax.plot(d.layer, d.control_shuffled_acc, color=MUTED, lw=1.2, ls=":", label="Shuffled-label control")
        ax.set_title(task.replace("_", " "), fontsize=10, color=INK)
        ax.set_xlabel("Layer", fontsize=9, color=MUTED)
        ax.set_ylim(0.0, 1.02)
        _style(ax)
    axes[0].set_ylabel("Accuracy (dashed = model's answer)", fontsize=9, color=MUTED)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower center", ncol=4)
    fig.suptitle(f"Probe accuracy by layer vs. the model's own answers ({model_name})", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_blindsight(summary, out, model_name):
    import matplotlib.pyplot as plt
    s = summary[summary.condition == "all"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(s))
    err = np.vstack([s.blindsight_rate - s.blindsight_ci_lo, s.blindsight_ci_hi - s.blindsight_rate])
    ax.bar(x, s.blindsight_rate, width=0.55, color=SERIES[0])
    ax.errorbar(x, s.blindsight_rate, yerr=err, fmt="none", ecolor=INK, lw=1.2, capsize=4)
    ax.axhline(0.5, color=MUTED, lw=1.2, ls="--")
    ax.text(len(s) - 0.5, 0.51, "chance", fontsize=8, color=MUTED, ha="right", va="bottom")
    for i, r in s.iterrows():
        ax.text(i, 0.02, f"n={r.n_report_wrong}", ha="center", fontsize=8, color="white")
    ax.set_xticks(x, [t.replace("_", "\n") for t in s.task], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(probe right | answer wrong)", fontsize=9, color=MUTED)
    ax.set_title(f"Blindsight rate: the probe still knows when the model answers wrong ({model_name})",
                 fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_confidence(df, out, model_name):
    import matplotlib.pyplot as plt
    groups = [
        ("Correct answer", df[df.report_correct == 1], SERIES[0]),
        ("Blindsight (probe right, answer wrong)", df[df.blindsight == 1], SERIES[1]),
        ("Both wrong", df[(df.report_correct == 0) & (df.probe_correct == 0)], SERIES[2]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bins = np.linspace(0.5, 1.0, 26)
    for name, g, c in groups:
        if len(g):
            ax.hist(g.report_conf, bins=bins, density=True, histtype="step", lw=2, color=c,
                    label=f"{name} (n={len(g)})")
    ax.set_xlabel("Model's confidence in its own answer", fontsize=9, color=MUTED)
    ax.set_ylabel("Density", fontsize=9, color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title(f"Confidence on blindsight items vs. ordinary items ({model_name})", fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_steering(steer_df, out, model_name):
    import matplotlib.pyplot as plt
    tasks = list(dict.fromkeys(steer_df.task))
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.4 * len(tasks), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, task in zip(axes, tasks):
        d = steer_df[steer_df.task == task]
        for kind, c, ls in (("probe", SERIES[0], "-"), ("random", MUTED, "--")):
            k = d[d.direction == kind].sort_values("alpha")
            ax.plot(k.alpha, k.wrong_to_right, color=c, lw=2, ls=ls, marker="o", ms=4,
                    label=f"{kind} direction: wrong→right")
            ax.plot(k.alpha, k.right_to_wrong, color=c, lw=1.2, ls=":", label=f"{kind} direction: right→wrong")
        ax.set_title(task.replace("_", " "), fontsize=10, color=INK)
        ax.set_xlabel("Steering strength α (× mean residual norm)", fontsize=8, color=MUTED)
        _style(ax)
    axes[0].set_ylabel("Share of items flipped", fontsize=9, color=MUTED)
    axes[-1].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(f"Steering along the probe's own prediction (H4, {model_name})", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
