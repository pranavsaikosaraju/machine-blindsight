"""Cross-model summary figures from results/*_all_models.csv.

    python scripts/make_summary_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "results"
OUT = ROOT / "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3df"
TASKS = ["number_compare", "city_country", "negated_city", "addition_check"]
TASK_NAMES = {"number_compare": "Number\ncomparison", "city_country": "City\nfacts",
              "negated_city": "Negated\ncity facts", "addition_check": "Addition\ncheck"}
MODELS = ["qwen2.5-1.5b-instruct", "qwen2.5-3b-instruct"]
MODEL_NAMES = {"qwen2.5-1.5b-instruct": "Qwen2.5-1.5B-Instruct", "qwen2.5-3b-instruct": "Qwen2.5-3B-Instruct"}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_answer_vs_probe(s):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, m in zip(axes, MODELS):
        d = s[s.model == m].set_index("task").loc[TASKS]
        x = np.arange(len(TASKS))
        ax.bar(x - 0.2, d.report_acc, 0.38, color=SERIES[1], label="Model's spoken answer")
        ax.bar(x + 0.2, d.probe_acc, 0.38, color=SERIES[0], label="Linear probe on activations")
        ax.axhline(0.5, color=MUTED, lw=1, ls="--")
        ax.set_xticks(x, [TASK_NAMES[t] for t in TASKS])
        ax.set_ylim(0, 1.05)
        ax.set_title(MODEL_NAMES[m], fontsize=10, color=INK)
        style(ax)
    axes[0].set_ylabel("Accuracy (all conditions)", fontsize=9, color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("The activations know the answer more often than the model says it", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "answer_vs_probe.png", dpi=160)
    plt.close(fig)


def fig_blindsight(s):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(TASKS))
    for i, m in enumerate(MODELS):
        d = s[s.model == m].set_index("task").loc[TASKS]
        off = (i - 0.5) * 0.38
        err = np.vstack([d.blindsight_rate - d.blindsight_ci_lo, d.blindsight_ci_hi - d.blindsight_rate])
        ax.bar(x + off, d.blindsight_rate, 0.36, color=[SERIES[0], SERIES[2]][i], label=MODEL_NAMES[m])
        ax.errorbar(x + off, d.blindsight_rate, yerr=err, fmt="none", ecolor=INK, lw=1, capsize=3)
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.text(3.6, 0.51, "chance", fontsize=8, color=MUTED, ha="right", va="bottom")
    ax.set_xticks(x, [TASK_NAMES[t] for t in TASKS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(probe right | answer wrong)", fontsize=9, color=MUTED)
    ax.set_title("Blindsight rate with 95% bootstrap CI", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "blindsight_rate.png", dpi=160)
    plt.close(fig)


def fig_steering(st):
    tasks = ["number_compare", "city_country", "negated_city"]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharey=True)
    for r, m in enumerate(MODELS):
        for c, t in enumerate(tasks):
            ax = axes[r, c]
            d = st[(st.model == m) & (st.task == t)]
            for kind, col, ls in (("probe", SERIES[0], "-"), ("random", MUTED, "--")):
                k = d[d.direction == kind].sort_values("alpha")
                ax.plot(k.alpha, k.wrong_to_right, color=col, lw=2, ls=ls, marker="o", ms=4,
                        label=f"{kind}: wrong→right")
                ax.plot(k.alpha, k.right_to_wrong, color=col, lw=1, ls=":", marker="x", ms=4,
                        label=f"{kind}: right→wrong")
            ax.set_ylim(-0.03, 1.03)
            ax.set_xticks([0.25, 0.5, 1.0])
            if r == 0:
                ax.set_title(TASK_NAMES[t].replace("\n", " "), fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(f"{MODEL_NAMES[m]}\nshare flipped", fontsize=9, color=MUTED)
            if r == 1:
                ax.set_xlabel("α (× mean residual norm)", fontsize=9, color=MUTED)
            style(ax)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle("Steering along the probe's own prediction (H4): clean in 1.5B, unstable in 3B", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "steering.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    s = pd.read_csv(ROOT / "summary_all_models.csv")
    st = pd.read_csv(ROOT / "steering_all_models.csv")
    fig_answer_vs_probe(s)
    fig_blindsight(s)
    fig_steering(st)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
