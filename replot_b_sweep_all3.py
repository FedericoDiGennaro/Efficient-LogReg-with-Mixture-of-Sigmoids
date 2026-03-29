#!/usr/bin/env python3
"""Build the paper-ready multi-dataset B-sweep figure from saved results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def quantiles_iqr(arr):
    q25 = np.quantile(arr, 0.25, axis=0)
    q50 = np.quantile(arr, 0.50, axis=0)
    q75 = np.quantile(arr, 0.75, axis=0)
    return q25, q50, q75


def load_payload(path):
    p = np.load(path, allow_pickle=False)
    return {
        "b_values": p["b_values"].astype(np.float64),
        "ew_steps": p["ew_steps"].astype(np.int64),
        "aioli_mat": p["aioli_mat"].astype(np.float64),
        "ew_mat": p["ew_mat"].astype(np.float64),
        "n": int(p["n"][0]),
        "repeats": int(p["repeats"][0]),
        "dataset_name": str(p["dataset_name"][0]),
        "dataset_split": str(p["dataset_split"][0]),
    }


def main():
    out_dir = Path("figs_online_mala_comparable")
    datasets = ["a9a", "w8a", "ijcnn1"]
    split = "train"
    n = 1000

    loaded = []
    for ds in datasets:
        npz_path = out_dir / f"{ds}_{split}_B_sweep_aioli_ew_n{n}_results.npz"
        if not npz_path.exists():
            print(f"[skip] missing {npz_path}")
            continue
        loaded.append(load_payload(npz_path))

    if not loaded:
        print("[skip] no B-sweep result files found")
        return

    # Match the large-font style used in previous replot scripts.
    TITLE_FS = 70
    XLABEL_FS = 45
    YLABEL_FS = 45
    TICK_FS = 42
    LEGEND_FS = 42
    LINE_W = 5
    FILL_ALPHA = 0.2

    fig, axes = plt.subplots(1, len(loaded), figsize=(50.0, 13.0), sharex=False, sharey=False)
    if len(loaded) == 1:
        axes = [axes]

    h_aioli = None
    h_ew = None
    for i, obj in enumerate(loaded):
        ax = axes[i]
        b = obj["b_values"]
        aioli_q25, aioli_q50, aioli_q75 = quantiles_iqr(obj["aioli_mat"])
        ew_q25, ew_q50, ew_q75 = quantiles_iqr(obj["ew_mat"])

        (h_aioli,) = ax.plot(b, aioli_q50, marker="o", linewidth=LINE_W, label="AIOLI")
        ax.fill_between(b, aioli_q25, aioli_q75, alpha=FILL_ALPHA)
        (h_ew,) = ax.plot(b, ew_q50, marker="s", linewidth=LINE_W, label="EW")
        ax.fill_between(b, ew_q25, ew_q75, alpha=FILL_ALPHA)

        ax.set_title(obj["dataset_name"], fontsize=TITLE_FS, pad=14)
        ax.set_xlabel("B", fontsize=XLABEL_FS, labelpad=8)
        ax.tick_params(axis="both", labelsize=TICK_FS)

        if i == 0:
            ax.set_ylabel(f"Average log-loss", fontsize=YLABEL_FS)

    fig.legend(
        [h_aioli, h_ew],
        ["AIOLI", "EW"],
        loc="lower center",
        ncol=2,
        fontsize=LEGEND_FS,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=[0.03, 0.12, 1.0, 1.0])

    out_pdf = out_dir / f"all3_{split}_B_sweep_aioli_ew_n{n}.pdf"
    out_pgf = out_dir / f"all3_{split}_B_sweep_aioli_ew_n{n}.pgf"
    fig.savefig(out_pdf)
    fig.savefig(out_pgf)
    plt.close(fig)
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_pgf}")


if __name__ == "__main__":
    main()
