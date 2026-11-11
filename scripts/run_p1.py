"""Run the full machine-blindsight experiment for one model.

Example (Kaggle, GPU T4 x2):
    python scripts/run_p1.py --model Qwen/Qwen2.5-1.5B-Instruct --out results/qwen2.5-1.5b

Steps: build data -> run model (answers + activations) -> probes per layer ->
pick layer -> blindsight metrics -> steering test -> random-init control -> figures.
Every step writes to --out so a crashed session can be inspected.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import analysis, probes  # noqa: E402
from src.datasets import build_dataset  # noqa: E402
from src.model import load_model, run_items  # noqa: E402
from src.steering import make_hook_factory, random_unit_like  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-per-task", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alphas", default="0.25,0.5,1.0")
    p.add_argument("--steer-max-items", type=int, default=600,
                   help="per task: all answer-wrong items (up to half of this) plus answer-right items")
    p.add_argument("--skip-steer", action="store_true")
    p.add_argument("--random-init-control", action="store_true")
    p.add_argument("--save-acts", action="store_true", help="save the activation cache (large)")
    args = p.parse_args()

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    short = args.model.split("/")[-1]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. data -----------------------------------------------------------------
    df = build_dataset(args.n_per_task, args.seed)
    log(f"dataset: {len(df)} items")

    # 2. model: answers + activations -----------------------------------------------
    lm = load_model(args.model)
    log(f"loaded {args.model} (chat template: {lm.is_chat}, layers: {lm.model.config.num_hidden_layers})")
    cols, acts = run_items(lm, df, batch_size=args.batch_size)
    for k, v in cols.items():
        df[k] = v
    df.to_csv(out / "items_with_reports.csv", index=False)
    if args.save_acts:
        np.save(out / "acts.npy", acts)
    log("report accuracy by task/condition:\n" +
        df.assign(ok=(df.report == df.label)).groupby(["task", "condition"]).ok.mean().round(3).to_string())

    # 3. probes -----------------------------------------------------------------
    log("training probes (leave-one-template-out) ...")
    layer_df, oof = probes.probe_all_layers(acts, df, seed=args.seed)
    layer_df.to_csv(out / "probe_by_layer.csv", index=False)

    # 4. metrics ----------------------------------------------------------------
    best = analysis.choose_layers(layer_df)
    best = {t: max(int(L), 1) for t, L in best.items()}  # steering needs layer >= 1
    df = analysis.attach_probe(df, oof, best)
    summary = analysis.summarize(df, layer_df)
    df.to_csv(out / "items_scored.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    log("summary:\n" + summary[["task", "condition", "layer", "report_acc", "probe_acc",
                                 "n_report_wrong", "blindsight_rate", "conf_auroc"]].round(3).to_string())

    # 5. steering (H4) ----------------------------------------------------------
    steer_rows = []
    if not args.skip_steer:
        alphas = [float(a) for a in args.alphas.split(",")]
        rng = np.random.default_rng(args.seed)
        for task, sub in df.groupby("task", sort=False):
            L = best[task]
            X = np.nan_to_num(acts[sub.index.to_numpy(), L, :].astype(np.float32), posinf=0.0, neginf=0.0)
            direction = probes.mass_mean_direction(X, sub.label.to_numpy())
            mean_norm = float(np.linalg.norm(X.astype(np.float32), axis=1).mean())
            wrong = sub[sub.report_correct == 0]
            right = sub[sub.report_correct == 1]
            k_wrong = min(len(wrong), args.steer_max_items // 2)
            k_right = min(len(right), args.steer_max_items - k_wrong)
            pick = pd.concat([wrong.sample(k_wrong, random_state=args.seed) if k_wrong else wrong.iloc[:0],
                              right.sample(k_right, random_state=args.seed)])
            pick = pick.reset_index(drop=True)
            signs = np.where(pick.probe_pred.to_numpy() == 1, 1.0, -1.0)
            for kind, vec in (("probe", direction), ("random", random_unit_like(direction, int(rng.integers(1e6))))):
                for a in alphas:
                    hook = make_hook_factory(lm, L, vec, signs, scale=a * mean_norm)
                    cols2, _ = run_items(lm, pick, batch_size=args.batch_size, hook_fn=hook,
                                         cache=False, progress=False)
                    new_ok = cols2["report"] == pick.label.to_numpy()
                    was_ok = pick.report_correct.to_numpy() == 1
                    bs = (pick.blindsight == 1).to_numpy()
                    steer_rows.append({
                        "task": task, "layer": L, "direction": kind, "alpha": a,
                        "n_wrong": int((~was_ok).sum()), "n_right": int(was_ok.sum()),
                        "wrong_to_right": float(new_ok[~was_ok].mean()) if (~was_ok).any() else np.nan,
                        "right_to_wrong": float((~new_ok)[was_ok].mean()) if was_ok.any() else np.nan,
                        "blindsight_fixed": float(new_ok[bs].mean()) if bs.any() else np.nan,
                        "non_blindsight_wrong_fixed": float(new_ok[~was_ok & ~bs].mean())
                        if (~was_ok & ~bs).any() else np.nan,
                    })
                    log(f"steer {task} L{L} {kind} a={a}: wrong->right {steer_rows[-1]['wrong_to_right']:.3f}, "
                        f"right->wrong {steer_rows[-1]['right_to_wrong']:.3f}")
        steer_df = pd.DataFrame(steer_rows)
        for t in steer_df.task.unique():  # alpha 0 = no change, by definition
            for kind in ("probe", "random"):
                steer_df.loc[len(steer_df)] = {"task": t, "direction": kind, "alpha": 0.0,
                                               "wrong_to_right": 0.0, "right_to_wrong": 0.0}
        steer_df.to_csv(out / "steering.csv", index=False)

    # 6. random-init control ------------------------------------------------------
    rand_rows = []
    if args.random_init_control:
        del lm
        gc.collect()
        torch.cuda.empty_cache()
        log("random-init control: same architecture, random weights ...")
        lm_r = load_model(args.model, random_init=True)
        _, acts_r = run_items(lm_r, df, batch_size=args.batch_size)
        for task, sub in df.groupby("task", sort=False):
            L = best[task]
            s = probes.oof_scores(acts_r[sub.index.to_numpy(), L, :].astype(np.float32),
                                  sub.label.to_numpy(), sub.template.to_numpy(), seed=args.seed)
            rand_rows.append({"task": task, "layer": L,
                              "random_init_probe_acc": float(((s > 0).astype(int) == sub.label).mean())})
        pd.DataFrame(rand_rows).to_csv(out / "random_init_control.csv", index=False)
        log(f"random-init control: {rand_rows}")

    # 7. figures + json -----------------------------------------------------------
    analysis.fig_layers(layer_df, summary, out / "figures" / "probe_by_layer.png", short)
    analysis.fig_blindsight(summary, out / "figures" / "blindsight_rate.png", short)
    analysis.fig_confidence(df, out / "figures" / "confidence.png", short)
    if steer_rows:
        analysis.fig_steering(steer_df, out / "figures" / "steering.png", short)
    meta = {"model": args.model, "n_items": len(df), "best_layers": best,
            "n_layers": int(acts.shape[1] - 1), "seed": args.seed, "alphas": args.alphas,
            "random_init_control": rand_rows}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
