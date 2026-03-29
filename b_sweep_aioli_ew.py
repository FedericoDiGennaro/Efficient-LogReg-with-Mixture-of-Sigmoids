#!/usr/bin/env python3
"""
B-sweep experiment for AIOLI and EW (MALA).

This submission version only saves the numeric results needed to recreate the
final B-sweep figure. Paper-ready plotting is handled by
`replot_b_sweep_all3.py`.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from a9a_experiment import AIOLI, GaussianEW_MALA, logloss_from_prob
from libsvm_datasets import load_libsvm_binary


def parse_num_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def run_one_stream_for_b(X, y, R, B, seed, aioli_params, ew_params):
    n, d = X.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    Xs = X[perm]
    ys = y[perm]

    aioli = AIOLI(
        d=d,
        B=float(B),
        R=float(R),
        lam=float(1.0 / (B * B)),
        seed=seed + 10_000,
        **aioli_params,
    )
    ew = GaussianEW_MALA(
        d=d,
        B=float(B),
        rng=np.random.default_rng(seed + 20_000),
        **ew_params,
    )

    cum_aioli = 0.0
    cum_ew = 0.0
    for t in range(1, n + 1):
        x_t = Xs[t - 1]
        y_t = int(ys[t - 1])

        cum_aioli += logloss_from_prob(aioli.predict_prob(x_t), y_t)
        cum_ew += logloss_from_prob(ew.predict_prob(x_t), y_t)

        aioli.update(x_t, y_t)
        ew.update(Xs[:t], ys[:t])

    return float(cum_aioli / n), float(cum_ew / n)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep B and save AIOLI/EW final average log-loss for later replotting."
    )
    parser.add_argument("--dataset", type=str, default="a9a", choices=["a9a", "w8a", "ijcnn1"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-rounds", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--b-values", type=str, default="0.01,0.1,0.5,1,3,5,10,100")
    parser.add_argument("--b-cap", type=float, default=50.0)
    parser.add_argument("--aioli-t-inner", type=int, default=80)
    parser.add_argument("--ew-n-particles", type=int, default=12)
    parser.add_argument("--ew-step-size", type=float, default=1e-2)
    parser.add_argument("--ew-mala-steps", type=int, default=20)
    parser.add_argument(
        "--ew-mala-steps-max",
        type=int,
        default=300,
        help="Upper cap for B-adaptive MALA steps.",
    )
    parser.add_argument("--ew-grad-clip", type=float, default=50.0)
    parser.add_argument("--ew-mala-window", type=int, default=0, help="0 means full prefix.")
    parser.add_argument(
        "--ew-steps-list",
        type=str,
        default=None,
        help="Optional comma-separated explicit MALA steps per B value (same length as --b-values).",
    )
    parser.add_argument(
        "--load-results",
        type=str,
        default=None,
        help="Path to a saved .npz file to inspect without rerunning.",
    )
    args = parser.parse_args()

    out_dir = Path("figs_online_mala_comparable")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.load_results is not None:
        npz_path = Path(args.load_results)
        payload = np.load(npz_path, allow_pickle=False)
        b_values = payload["b_values"].astype(np.float64).tolist()
        aioli_mat = payload["aioli_mat"].astype(np.float64)
        ew_mat = payload["ew_mat"].astype(np.float64)
        n = int(payload["n"][0])
        dataset_name = str(payload["dataset_name"][0])
        dataset_split = str(payload["dataset_split"][0])
        ew_steps_list = payload["ew_steps"].astype(int).tolist()
        repeats = int(payload["repeats"][0])
        print(f"[load] {npz_path}")
        print(f"[load] dataset={dataset_name}/{dataset_split}, n={n}, repeats={repeats}")
    else:
        raw_b_values = parse_num_list(args.b_values)
        b_cap = float(args.b_cap)
        b_values = [min(float(b), b_cap) for b in raw_b_values]
        if b_values != raw_b_values:
            print(f"[grid] applied B cap={b_cap:g}: {raw_b_values} -> {b_values}")

        X, y, meta = load_libsvm_binary(
            args.dataset,
            split=args.split,
            data_dir="data",
            max_rounds=int(args.max_rounds),
            dense=True,
            add_intercept=True,
            scale_max_norm=True,
        )
        R = float(meta.R)
        n = int(meta.n)
        repeats = int(args.repeats)
        dataset_name = meta.name
        dataset_split = meta.split

        aioli_params = dict(T_inner=int(args.aioli_t_inner), omega_solver="lbfgs")
        ew_base_params = dict(
            n_particles=int(args.ew_n_particles),
            step_size=float(args.ew_step_size),
            grad_clip=float(args.ew_grad_clip),
            adapt_step=True,
            target_accept=0.57,
            rm_gamma0=0.5,
            rm_gamma_decay=0.25,
            step_size_min=1e-4,
            step_size_max=1.0,
            mala_window=None if int(args.ew_mala_window) <= 0 else int(args.ew_mala_window),
        )

        print(f"[data] dataset={meta.name}/{meta.split}, X={X.shape}, R={R:.3f}, rounds={n}, repeats={repeats}")
        print(f"[grid] B values={b_values}")

        aioli_mat = np.zeros((repeats, len(b_values)), dtype=np.float64)
        ew_mat = np.zeros((repeats, len(b_values)), dtype=np.float64)

        base_mala_steps = int(args.ew_mala_steps)
        max_mala_steps = int(args.ew_mala_steps_max)

        if args.ew_steps_list is not None:
            ew_steps_list = [int(round(v)) for v in parse_num_list(args.ew_steps_list)]
            if len(ew_steps_list) != len(b_values):
                raise ValueError(
                    f"--ew-steps-list length {len(ew_steps_list)} must match number of B values {len(b_values)}"
                )
            ew_steps_list = [int(np.clip(v, base_mala_steps, max_mala_steps)) for v in ew_steps_list]
        else:
            ew_steps_list = []
            for j, B in enumerate(b_values):
                ew_steps = int(np.ceil(base_mala_steps * np.sqrt(max(B, 1e-12))))
                ew_steps = int(np.clip(ew_steps, base_mala_steps, max_mala_steps))
                if j >= len(b_values) - 2:
                    ew_steps = int(np.clip(int(np.ceil(2.0 * ew_steps)), base_mala_steps, max_mala_steps))
                ew_steps_list.append(ew_steps)

        for j, B in enumerate(b_values):
            ew_params = dict(ew_base_params)
            ew_params["mala_steps_per_round"] = int(ew_steps_list[j])
            print(f"[B={B:g}] EW mala_steps_per_round={ew_steps_list[j]}")
            for r in range(repeats):
                print(f"[run] B={B:g} repeat {r + 1}/{repeats}")
                a_loss, e_loss = run_one_stream_for_b(
                    X,
                    y,
                    R=R,
                    B=float(B),
                    seed=int(args.seed0 + 1000 * j + r),
                    aioli_params=aioli_params,
                    ew_params=ew_params,
                )
                aioli_mat[r, j] = a_loss
                ew_mat[r, j] = e_loss
            print(
                "[B={:g}] AIOLI mean={:.6f}, EW mean={:.6f}".format(
                    B, float(aioli_mat[:, j].mean()), float(ew_mat[:, j].mean())
                )
            )

        base = f"{dataset_name}_{dataset_split}_B_sweep_aioli_ew_n{n}"
        npz_path = out_dir / f"{base}_results.npz"
        np.savez_compressed(
            npz_path,
            b_values=np.asarray(b_values, dtype=np.float64),
            ew_steps=np.asarray(ew_steps_list, dtype=np.int64),
            aioli_mat=aioli_mat,
            ew_mat=ew_mat,
            n=np.asarray([n], dtype=np.int64),
            repeats=np.asarray([repeats], dtype=np.int64),
            dataset_name=np.asarray([dataset_name]),
            dataset_split=np.asarray([dataset_split]),
        )
        meta_json = out_dir / f"{base}_meta.json"
        meta_payload = dict(
            dataset=dataset_name,
            split=dataset_split,
            n=n,
            repeats=repeats,
            b_values=b_values,
            ew_steps=ew_steps_list,
            cmd_vars=dict(
                aioli_t_inner=int(args.aioli_t_inner),
                ew_n_particles=int(args.ew_n_particles),
                ew_step_size=float(args.ew_step_size),
                ew_grad_clip=float(args.ew_grad_clip),
                ew_mala_window=int(args.ew_mala_window),
            ),
        )
        meta_json.write_text(json.dumps(meta_payload, indent=2))
        print(f"[saved] {npz_path}")
        print(f"[saved] {meta_json}")

    print("[summary] B\tAIOLI_mean\tEW_mean")
    for j, B in enumerate(b_values):
        print(f"[summary] {B:.6g}\t{float(aioli_mat[:, j].mean()):.6f}\t{float(ew_mat[:, j].mean()):.6f}")


if __name__ == "__main__":
    main()
