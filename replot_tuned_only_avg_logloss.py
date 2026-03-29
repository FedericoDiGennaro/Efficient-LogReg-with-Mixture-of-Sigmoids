#!/usr/bin/env python3
"""Build the paper-ready multi-dataset average log-loss figure from saved results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def aggregate_quantiles(mat):
    return (
        np.quantile(mat, 0.25, axis=0),
        np.quantile(mat, 0.50, axis=0),
        np.quantile(mat, 0.75, axis=0),
    )


def main():
    out_dir = Path("figs_online_mala_comparable")
    datasets = ["a9a", "w8a", "ijcnn1"]
    split = "train"
    mode = "unreg_ogd_ons"

    curve_targets = [
        ("OGD", "OGD"),
        ("ONS", "ONS"),
        ("AIOLI-Bopt", r"AIOLI (tuned $\lambda$)"),
        ("EW-Bopt", r"EW (tuned $\lambda$)"),
    ]

    loaded = []
    for ds in datasets:
        npz_path = out_dir / f"{ds}_{split}_{mode}_online_mala_comparable_results.npz"
        meta_path = out_dir / f"{ds}_{split}_{mode}_online_mala_comparable_meta.json"
        if not npz_path.exists() or not meta_path.exists():
            print(f"[skip] missing files for {ds}")
            continue
        payload = np.load(npz_path, allow_pickle=False)
        meta = json.loads(meta_path.read_text())
        key_map = meta["curve_key_map"]  # safe_key -> original key
        inv_map = {v: k for k, v in key_map.items()}  # original key -> safe_key
        loaded.append((ds, payload, inv_map))

    if not loaded:
        print("[skip] no dataset files found")
        return

    # ---- Styling knobs (adjust here) ----
    TITLE_FS = 70
    XLABEL_FS = 45
    YLABEL_FS = 45
    TICK_FS = 42
    LEGEND_FS = 42
    LINE_W = 5
    FILL_ALPHA = 0.18

    # Make figure taller: top row loss, bottom row acceptance/step-size.
    fig_w = 50.0
    fig_h = 23.0
    fig, axes = plt.subplots(
        2,
        len(loaded),
        figsize=(fig_w, fig_h),
        sharex="col",
        gridspec_kw={"height_ratios": [3.0, 1.35]},
    )
    if len(loaded) == 1:
        # Normalize shape to [2][1]
        axes = np.array([[axes[0]], [axes[1]]], dtype=object)

    method_handles, method_labels = [], []
    bottom_handles, bottom_labels = [], []

    for ax_i, (ds, payload, inv_map) in enumerate(loaded):
        ax_top = axes[0, ax_i]
        ax_bot = axes[1, ax_i]
        t = payload["t"]
        for j, (raw_key, label) in enumerate(curve_targets):
            if raw_key not in inv_map:
                raise KeyError(f"{ds}: missing curve key '{raw_key}' in inv_map. Available: {list(inv_map.keys())}")
            safe_key = inv_map[raw_key]
            mat = payload[safe_key]
            q25, q50, q75 = aggregate_quantiles(mat)

            line = ax_top.plot(t, q50, linewidth=LINE_W)[0]
            ax_top.fill_between(t, q25, q75, alpha=FILL_ALPHA)

            # Collect legend items only once (from first axis)
            if ax_i == 0:
                method_handles.append(line)
                method_labels.append(label)

        # Bottom panel: tuned EW acceptance + target + tuned EW step size.
        for req in ["EW-Bopt-accept", "EW-Bopt-h"]:
            if req not in inv_map:
                raise KeyError(f"{ds}: missing curve key '{req}' in inv_map. Available: {list(inv_map.keys())}")

        acc_mat = payload[inv_map["EW-Bopt-accept"]]
        aq25, aq50, aq75 = aggregate_quantiles(acc_mat)
        line_acc = ax_bot.plot(t, aq50, color="tab:brown", linewidth=2.2)[0]
        ax_bot.fill_between(t, aq25, aq75, color="tab:brown", alpha=0.2)

        line_target = ax_bot.axhline(0.57, color="tab:green", linestyle="--", linewidth=2.0)

        h_mat = payload[inv_map["EW-Bopt-h"]]
        hq25, hq50, hq75 = aggregate_quantiles(h_mat)
        line_h = ax_bot.plot(t, hq50, color="tab:cyan", linewidth=2.0)[0]
        ax_bot.fill_between(t, hq25, hq75, color="tab:cyan", alpha=0.18)

        if ax_i == 0:
            bottom_handles.extend([line_acc, line_target, line_h])
            bottom_labels.extend(
                ["acceptance rate EW (MALA)", "target acceptance", "step size MALA"]
            )

        ax_top.set_title(ds, fontsize=TITLE_FS, pad=14)
        ax_top.set_xscale("log")
        ax_top.tick_params(axis="both", labelsize=TICK_FS)

        ax_bot.set_xscale("log")
        ax_bot.set_ylim(0.0, 1.05)
        ax_bot.set_yticks([0.0, 0.5, 1.0])
        ax_bot.set_xlabel("t (round)", fontsize=XLABEL_FS, labelpad=10)
        ax_bot.tick_params(axis="both", labelsize=TICK_FS)

    # Keep ylabel only on the top row (left subplot), not global across both rows.
    axes[0, 0].set_ylabel("Average log-loss", fontsize=YLABEL_FS)

    # Single legend box, two rows: first row methods, second row bottom-panel quantities.
    # Matplotlib fills multi-column legends column-wise in this backend.
    # Build entries as [m1,b1,m2,b2,m3,b3,m4,blank] so rows appear grouped.
    blank = Line2D([], [], linewidth=0, alpha=0)
    legend_handles = [
        method_handles[0], bottom_handles[0],
        method_handles[1], bottom_handles[1],
        method_handles[2], bottom_handles[2],
        method_handles[3], blank,
    ]
    legend_labels = [
        method_labels[0], bottom_labels[0],
        method_labels[1], bottom_labels[1],
        method_labels[2], bottom_labels[2],
        method_labels[3], "",
    ]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        fontsize=LEGEND_FS,
        frameon=True,
        handlelength=3.0,
        columnspacing=1.4,
        handletextpad=0.8,
        borderpad=0.8,
        bbox_to_anchor=(0.5, 0.0),
    )

    # Leave enough room at the bottom for the combined legend box
    fig.tight_layout(rect=[0.03, 0.14, 1.0, 1.0])

    out_pdf = out_dir / f"all3_{split}_{mode}_online_avg_logloss_tuned_only.pdf"
    out_pgf = out_dir / f"all3_{split}_{mode}_online_avg_logloss_tuned_only.pgf"
    fig.savefig(out_pdf)
    fig.savefig(out_pgf)
    plt.close(fig)
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_pgf}")


if __name__ == "__main__":
    main()
