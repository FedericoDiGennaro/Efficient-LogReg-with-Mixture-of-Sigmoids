#!/usr/bin/env python3
"""
Comparable online logistic baselines on LIBSVM binary datasets.

This submission version keeps only the pieces needed to reproduce the
final average log-loss figure. The script:

1. loads one dataset,
2. tunes AIOLI's lambda and EW's Gaussian-prior radius B on log-loss only,
3. runs repeated prequential experiments,
4. saves the raw trajectories to `figs_online_mala_comparable/*.npz`,
5. saves metadata to `figs_online_mala_comparable/*.json`.

Paper-ready figures are produced separately by `replot_tuned_only_avg_logloss.py`.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, solve_triangular
from scipy.optimize import minimize
from scipy.special import expit

from libsvm_datasets import load_libsvm_binary


def logloss_from_prob(p1, y_pm1):
    p1 = np.clip(p1, 1e-12, 1.0 - 1e-12)
    return float(-np.log(p1) if y_pm1 == 1 else -np.log(1.0 - p1))


def grad_reg_logistic(w, x, y_pm1, B):
    z = float(w @ x)
    s = expit(-y_pm1 * z)
    grad_lik = (-y_pm1 * s) * x
    grad_reg = w / (B**2)
    return grad_lik + grad_reg


def grad_logistic_only(w, x, y_pm1):
    z = float(w @ x)
    s = expit(-y_pm1 * z)
    return (-y_pm1 * s) * x


def clip_by_norm(v, max_norm):
    if max_norm <= 0:
        return v
    nrm = np.linalg.norm(v)
    if nrm <= max_norm:
        return v
    return v * (max_norm / (nrm + 1e-12))


def cholupdate_lower(L, x):
    """Rank-1 Cholesky update for lower-triangular L where A = L L^T."""
    L = L.copy()
    x = x.copy()
    n = L.shape[0]
    for k in range(n):
        r = np.sqrt(L[k, k] ** 2 + x[k] ** 2)
        c = r / (L[k, k] + 1e-18)
        s = x[k] / (L[k, k] + 1e-18)
        L[k, k] = r
        if k + 1 < n:
            L[k + 1 :, k] = (L[k + 1 :, k] + s * x[k + 1 :]) / (c + 1e-18)
            x[k + 1 :] = c * x[k + 1 :] - s * L[k + 1 :, k]
    return L


class AIOLI:
    """AIOLI (Algorithm 2, Jezequel et al., COLT 2020) for binary logistic."""

    def __init__(
        self,
        d,
        B,
        R,
        lam=None,
        T_inner=50,
        seed=0,
        omega_solver="lbfgs",
        omega_gd_alpha=None,
        omega_gd_steps=None,
        omega_lbfgs_gtol=1e-10,
        omega_lbfgs_ftol=1e-12,
    ):
        self.d = int(d)
        self.B = float(B)
        self.R = float(R)
        self.lam = float(1.0 / (B * B) if lam is None else lam)
        self.T_inner = int(T_inner)

        self.L = (self.lam ** (-0.5)) * np.eye(self.d, dtype=np.float64)
        self.b = np.zeros(self.d, dtype=np.float64)
        self.theta = np.zeros(self.d, dtype=np.float64)

        self.alpha = self.lam / (4.0 * self.lam + self.R * self.R)
        self.rng = np.random.default_rng(seed)
        self.omega_solver = str(omega_solver).lower().strip()
        self.omega_gd_alpha = float(self.alpha if omega_gd_alpha is None else omega_gd_alpha)
        self.omega_gd_steps = int(self.T_inner if omega_gd_steps is None else omega_gd_steps)
        self.omega_lbfgs_gtol = float(omega_lbfgs_gtol)
        self.omega_lbfgs_ftol = float(omega_lbfgs_ftol)

    @staticmethod
    def _omega_obj_grad(omega, u, v):
        a = float(v @ omega)
        obj = float(
            omega @ omega
            - 2.0 * (u @ omega)
            + np.logaddexp(0.0, -a)
            + np.logaddexp(0.0, a)
        )
        grad = 2.0 * omega - 2.0 * u + (-expit(-a) + expit(a)) * v
        return obj, grad

    def _solve_omega(self, u, v):
        def f(w):
            val, _ = self._omega_obj_grad(w, u, v)
            return val

        def g(w):
            _, gg = self._omega_obj_grad(w, u, v)
            return gg

        if self.omega_solver == "lbfgs":
            res = minimize(
                fun=f,
                x0=np.zeros_like(u),
                jac=g,
                method="L-BFGS-B",
                options=dict(
                    maxiter=self.T_inner,
                    gtol=self.omega_lbfgs_gtol,
                    ftol=self.omega_lbfgs_ftol,
                ),
            )
            return res.x.astype(np.float64, copy=False)

        if self.omega_solver == "gd":
            omega = np.zeros_like(u)
            for _ in range(self.omega_gd_steps):
                _, grad = self._omega_obj_grad(omega, u, v)
                omega = omega - self.omega_gd_alpha * grad
            return omega

        raise ValueError(f"Unknown omega_solver={self.omega_solver}. Use 'lbfgs' or 'gd'.")

    def _compute_theta(self, x):
        z_b = solve_triangular(self.L, self.b, lower=True, check_finite=False)
        z_x = solve_triangular(self.L, x, lower=True, check_finite=False)
        W = np.stack([z_b, z_x], axis=1)

        M = W.T @ W
        evals, U = eigh(M)
        idx = np.where(evals > 1e-12)[0]
        if idx.size == 0:
            self.theta = np.zeros(self.d, dtype=np.float64)
            return

        evals = evals[idx]
        U = U[:, idx]
        sig_sqrt = np.sqrt(evals)
        sig_inv_sqrt = 1.0 / sig_sqrt

        e1 = np.array([1.0, 0.0], dtype=np.float64)
        e2 = np.array([0.0, 1.0], dtype=np.float64)
        u = sig_sqrt * (U.T @ e1)
        v = sig_sqrt * (U.T @ e2)

        omega = self._solve_omega(u, v)
        q = U @ (sig_inv_sqrt * omega)
        r = W @ q
        self.theta = solve_triangular(self.L.T, r, lower=False, check_finite=False)

    def predict_prob(self, x):
        self._compute_theta(x)
        return float(expit(self.theta @ x))

    def update(self, x, y_pm1):
        self._compute_theta(x)
        yhat = float(self.theta @ x)

        denom = 1.0 + np.exp(np.clip(y_pm1 * yhat, -50.0, 50.0))
        g = (-y_pm1 / denom) * x
        eta = float(np.exp(np.clip(y_pm1 * yhat, -50.0, 50.0))) / (1.0 + self.B * self.R)

        update_vec = np.sqrt(max(eta, 0.0) / 2.0) * g
        self.L = cholupdate_lower(self.L, update_vec)
        self.b = self.b + (eta * float(g @ self.theta) - 1.0) * g


class OGD_L2:
    def __init__(self, d, B=5.0, eta=0.5, grad_clip=0.0, regularized=True):
        self.w = np.zeros(d, dtype=np.float64)
        self.B = float(B)
        self.eta = float(eta)
        self.grad_clip = float(grad_clip)
        self.regularized = bool(regularized)

    def predict_prob(self, x):
        return float(expit(self.w @ x))

    def update(self, x, y):
        if self.regularized:
            g = grad_reg_logistic(self.w, x, y, self.B)
        else:
            g = grad_logistic_only(self.w, x, y)
        g = clip_by_norm(g, self.grad_clip)
        self.w = self.w - self.eta * g


class ONS_L2:
    def __init__(self, d, B=5.0, eta=1.0, eps=1.0, grad_clip=0.0, regularized=True):
        self.w = np.zeros(d, dtype=np.float64)
        self.B = float(B)
        self.eta = float(eta)
        self.grad_clip = float(grad_clip)
        self.regularized = bool(regularized)
        self.A_inv = (1.0 / float(eps)) * np.eye(d, dtype=np.float64)

    def predict_prob(self, x):
        return float(expit(self.w @ x))

    def update(self, x, y):
        if self.regularized:
            g = grad_reg_logistic(self.w, x, y, self.B)
        else:
            g = grad_logistic_only(self.w, x, y)
        g = clip_by_norm(g, self.grad_clip)

        v = self.A_inv @ g
        denom = 1.0 + float(g @ v)
        self.A_inv = self.A_inv - np.outer(v, v) / denom

        step = self.A_inv @ g
        self.w = self.w - self.eta * step


class GaussianEW_MALA:
    """
    EW with Gaussian prior via MALA.

    rho_t(theta) is proportional to the prefix logistic likelihood times the
    Gaussian prior N(0, B^2 I).
    """

    def __init__(
        self,
        d,
        B=5.0,
        n_particles=12,
        step_size=1e-2,
        mala_steps_per_round=20,
        grad_clip=50.0,
        adapt_step=True,
        mala_window=None,
        rng=None,
        target_accept=0.57,
        rm_gamma0=0.05,
        rm_gamma_decay=0.5,
        step_size_min=1e-4,
        step_size_max=2e-1,
    ):
        self.d = int(d)
        self.B = float(B)
        self.S = int(n_particles)
        self.h = float(step_size)
        self.log_h = float(np.log(self.h))
        self.K = int(mala_steps_per_round)
        self.grad_clip = float(grad_clip)
        self.adapt_step = bool(adapt_step)
        self.target_accept = float(target_accept)
        self.mala_window = mala_window
        self.rng = np.random.default_rng() if rng is None else rng
        self.rm_gamma0 = float(rm_gamma0)
        self.rm_gamma_decay = float(rm_gamma_decay)
        self.h_min = float(step_size_min)
        self.h_max = float(step_size_max)
        self.round = 0

        self.thetas = self.rng.normal(scale=self.B, size=(self.S, self.d)).astype(np.float64)

    def predict_prob(self, x):
        return float(expit(self.thetas @ x).mean())

    def _U_and_grad(self, thetas, X, y):
        Z = X @ thetas.T
        yz = y[:, None] * Z

        U_lik = np.sum(np.logaddexp(0.0, -yz), axis=0)
        U_prior = 0.5 * np.sum(thetas * thetas, axis=1) / (self.B**2)
        U = U_lik + U_prior

        sig = expit(-yz)
        W = (-y)[:, None] * sig
        grad_lik = (X.T @ W).T
        grad_prior = thetas / (self.B**2)
        grad = grad_lik + grad_prior

        if self.grad_clip > 0:
            norms = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
            grad = grad * np.minimum(1.0, self.grad_clip / norms)

        return U, grad

    def _mala_step(self, X, y):
        h = self.h
        S, d = self.thetas.shape

        U, g = self._U_and_grad(self.thetas, X, y)
        mean_fwd = self.thetas - 0.5 * h * g
        prop = mean_fwd + np.sqrt(h) * self.rng.normal(size=(S, d))

        U_p, g_p = self._U_and_grad(prop, X, y)
        mean_bwd = prop - 0.5 * h * g_p

        logq_fwd = -np.sum((prop - mean_fwd) ** 2, axis=1) / (2.0 * h)
        logq_bwd = -np.sum((self.thetas - mean_bwd) ** 2, axis=1) / (2.0 * h)

        log_alpha = -(U_p - U) + (logq_bwd - logq_fwd)
        accept = np.log(self.rng.uniform(size=S)) < log_alpha

        self.thetas[accept] = prop[accept]
        return float(np.mean(accept))

    def _adapt_stepsize(self, acc_mean):
        self.round += 1
        gamma_t = self.rm_gamma0 / (self.round ** self.rm_gamma_decay)
        self.log_h += gamma_t * (acc_mean - self.target_accept)
        self.log_h = float(np.clip(self.log_h, np.log(self.h_min), np.log(self.h_max)))
        self.h = float(np.exp(self.log_h))

    def update(self, X_prefix, y_prefix):
        if self.mala_window is not None and self.mala_window > 0:
            X_use = X_prefix[-self.mala_window :]
            y_use = y_prefix[-self.mala_window :]
        else:
            X_use = X_prefix
            y_use = y_prefix

        accs = []
        for _ in range(self.K):
            acc = self._mala_step(X_use, y_use)
            accs.append(acc)
            if self.adapt_step:
                self._adapt_stepsize(acc)

        acc_mean = float(np.mean(accs)) if accs else 0.0
        return acc_mean, self.h


def run_one_permutation(
    X,
    y,
    B,
    R,
    ogd_params,
    ons_params,
    aioli_opt_B,
    aioli_opt_lam,
    ew_opt_B,
    aioli_shared_params,
    ew_params,
    seed,
):
    n, d = X.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    X_stream = X[perm]
    y_stream = y[perm]

    ogd = OGD_L2(d, B=B, **ogd_params)
    ons = ONS_L2(d, B=B, **ons_params)
    aioli_opt = AIOLI(
        d=d,
        B=float(aioli_opt_B),
        R=R,
        lam=float(aioli_opt_lam),
        seed=seed + 20_000,
        **aioli_shared_params,
    )
    aioli_b5 = AIOLI(
        d=d,
        B=float(B),
        R=R,
        lam=float(1.0 / (B * B)),
        seed=seed + 21_000,
        **aioli_shared_params,
    )
    ew_b5 = GaussianEW_MALA(d, B=float(B), rng=np.random.default_rng(seed + 999), **ew_params)
    ew_opt = GaussianEW_MALA(d, B=float(ew_opt_B), rng=np.random.default_rng(seed + 1999), **ew_params)

    names = ["OGD", "ONS", "AIOLI-Bopt", "AIOLI-B5", "EW-B5", "EW-Bopt"]
    avg_losses = {name: np.zeros(n, dtype=np.float64) for name in names}
    cum = {name: 0.0 for name in names}

    acc_track_b5 = np.zeros(n, dtype=np.float64)
    h_track_b5 = np.zeros(n, dtype=np.float64)
    acc_track_bopt = np.zeros(n, dtype=np.float64)
    h_track_bopt = np.zeros(n, dtype=np.float64)

    for t in range(1, n + 1):
        x_t = X_stream[t - 1]
        y_t = int(y_stream[t - 1])

        round_probs = {
            "OGD": ogd.predict_prob(x_t),
            "ONS": ons.predict_prob(x_t),
            "AIOLI-Bopt": aioli_opt.predict_prob(x_t),
            "AIOLI-B5": aioli_b5.predict_prob(x_t),
            "EW-B5": ew_b5.predict_prob(x_t),
            "EW-Bopt": ew_opt.predict_prob(x_t),
        }
        for name in names:
            cum[name] += logloss_from_prob(round_probs[name], y_t)
            avg_losses[name][t - 1] = cum[name] / t

        ogd.update(x_t, y_t)
        ons.update(x_t, y_t)
        aioli_opt.update(x_t, y_t)
        aioli_b5.update(x_t, y_t)

        X_prefix = X_stream[:t]
        y_prefix = y_stream[:t]
        acc_b5, h_b5 = ew_b5.update(X_prefix, y_prefix)
        acc_bopt, h_bopt = ew_opt.update(X_prefix, y_prefix)
        acc_track_b5[t - 1] = acc_b5
        h_track_b5[t - 1] = h_b5
        acc_track_bopt[t - 1] = acc_bopt
        h_track_bopt[t - 1] = h_bopt

    avg_losses["EW-B5-accept"] = acc_track_b5
    avg_losses["EW-B5-h"] = h_track_b5
    avg_losses["EW-Bopt-accept"] = acc_track_bopt
    avg_losses["EW-Bopt-h"] = h_track_bopt
    return avg_losses


def parse_num_list(s, cast=float):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def run_aioli_only_stream(X, y, B, R, aioli_kwargs, seed=0):
    n, d = X.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    Xs = X[perm]
    ys = y[perm]

    aioli = AIOLI(d=d, B=B, R=R, seed=seed + 1234, **aioli_kwargs)

    cum_ll = 0.0
    for t in range(n):
        p = aioli.predict_prob(Xs[t])
        cum_ll += logloss_from_prob(p, int(ys[t]))
        aioli.update(Xs[t], int(ys[t]))

    return float(cum_ll / n)


def tune_aioli_lambda(X, y, B, R, aioli_base_kwargs, lam_grid, seed=0):
    rows = []
    for lam in lam_grid:
        kwargs = dict(aioli_base_kwargs)
        kwargs["lam"] = float(lam)
        val = run_aioli_only_stream(X, y, B=B, R=R, aioli_kwargs=kwargs, seed=seed)
        rows.append((float(lam), float(val)))
    rows.sort(key=lambda x: x[1])
    return rows[0][0], rows


def run_ew_only_stream(X, y, B, ew_kwargs, seed=0, tune_rounds=None):
    n, d = X.shape
    n_use = n if tune_rounds is None else min(int(tune_rounds), n)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    Xs = X[perm][:n_use]
    ys = y[perm][:n_use]

    ew = GaussianEW_MALA(
        d=d,
        B=float(B),
        rng=np.random.default_rng(seed + 4321),
        **ew_kwargs,
    )

    cum_ll = 0.0
    for t in range(1, n_use + 1):
        p = ew.predict_prob(Xs[t - 1])
        cum_ll += logloss_from_prob(p, int(ys[t - 1]))
        ew.update(Xs[:t], ys[:t])
    return float(cum_ll / n_use)


def tune_ew_b(X, y, ew_base_kwargs, b_grid, seed=0, tune_rounds=None):
    rows = []
    for b in b_grid:
        val = run_ew_only_stream(
            X,
            y,
            B=float(b),
            ew_kwargs=ew_base_kwargs,
            seed=seed,
            tune_rounds=tune_rounds,
        )
        rows.append((float(b), float(val)))
    rows.sort(key=lambda x: x[1])
    return rows[0][0], rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submission-sized experiment driver for the final average log-loss comparison."
    )
    parser.add_argument("--dataset", type=str, default="a9a", choices=["a9a", "w8a", "ijcnn1"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-rounds", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--unregularized-ogd-ons",
        action="store_true",
        help="If set, OGD/ONS use unregularized logistic gradients (EW/AIOLI unchanged).",
    )
    parser.add_argument(
        "--aioli-lam",
        type=float,
        default=None,
        help="If set, use this fixed AIOLI lambda and skip tuning.",
    )
    parser.add_argument(
        "--aioli-lam-grid",
        type=str,
        default="0.0025,0.01,0.04,0.16,0.64,2.56,10.24",
        help="Comma-separated lambda grid for AIOLI tuning.",
    )
    parser.add_argument(
        "--aioli-lam-tune-seed",
        type=int,
        default=0,
        help="Seed used for the AIOLI lambda tuning permutation.",
    )
    parser.add_argument(
        "--ew-b",
        type=float,
        default=None,
        help="If set, use this fixed EW radius B and skip EW-B tuning.",
    )
    parser.add_argument(
        "--ew-b-grid",
        type=str,
        default="0.01,0.1,0.5,1,2,5,8,10",
        help="Comma-separated B grid for EW-B tuning.",
    )
    parser.add_argument(
        "--ew-b-tune-seed",
        type=int,
        default=0,
        help="Seed used for EW-B tuning permutation.",
    )
    parser.add_argument(
        "--ew-b-tune-rounds",
        type=int,
        default=500,
        help="Number of rounds used for EW-B tuning (<= max-rounds).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    B = 5.0
    ogd_params = dict(eta=0.5, grad_clip=0.0)
    ons_params = dict(eta=1.0, eps=1.0, grad_clip=0.0)
    aioli_shared_params = dict(T_inner=80, omega_solver="lbfgs")
    ew_params = dict(
        n_particles=12,
        step_size=1e-2,
        mala_steps_per_round=20,
        grad_clip=50.0,
        adapt_step=True,
        target_accept=0.57,
        rm_gamma0=0.5,
        rm_gamma_decay=0.25,
        step_size_min=1e-4,
        step_size_max=1.0,
        mala_window=None,
    )

    repeats = int(args.repeats)
    max_rounds = int(args.max_rounds)
    seed0 = 0
    dataset_name = args.dataset
    dataset_split = args.split
    unreg_ogd_ons = bool(args.unregularized_ogd_ons)
    aioli_lam_fixed = args.aioli_lam
    aioli_lam_grid = parse_num_list(args.aioli_lam_grid, cast=float)
    aioli_lam_tune_seed = int(args.aioli_lam_tune_seed)
    ew_b_fixed = args.ew_b
    ew_b_grid = parse_num_list(args.ew_b_grid, cast=float)
    ew_b_tune_seed = int(args.ew_b_tune_seed)
    ew_b_tune_rounds = int(args.ew_b_tune_rounds)

    out_dir = Path("figs_online_mala_comparable")
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, meta = load_libsvm_binary(
        dataset_name,
        split=dataset_split,
        data_dir="data",
        max_rounds=max_rounds,
        dense=True,
        add_intercept=True,
        scale_max_norm=True,
    )
    R = float(meta.R)
    n = meta.n

    if aioli_lam_fixed is not None:
        aioli_lam_star = float(aioli_lam_fixed)
        lam_rows = [(aioli_lam_star, float("nan"))]
        print(f"[AIOLI-tune] fixed lambda={aioli_lam_star:.6g}")
    else:
        aioli_lam_star, lam_rows = tune_aioli_lambda(
            X,
            y,
            B=B,
            R=R,
            aioli_base_kwargs=aioli_shared_params,
            lam_grid=aioli_lam_grid,
            seed=aioli_lam_tune_seed,
        )
        print(
            "[AIOLI-tune] best lambda={:.6g} from grid {}".format(
                aioli_lam_star, [float(v) for v in aioli_lam_grid]
            )
        )
        print("[AIOLI-tune] lambda\tavg_logloss")
        for lam, ll in lam_rows:
            print(f"[AIOLI-tune] {lam:.6g}\t{ll:.6f}")

    aioli_opt_B = float(1.0 / np.sqrt(max(aioli_lam_star, 1e-12)))
    aioli_b5_lam = float(1.0 / (B * B))

    if ew_b_fixed is not None:
        ew_opt_B = float(ew_b_fixed)
        ew_rows = [(ew_opt_B, float("nan"))]
        print(f"[EW-tune] fixed B={ew_opt_B:.6g}")
    else:
        ew_opt_B, ew_rows = tune_ew_b(
            X,
            y,
            ew_base_kwargs=ew_params,
            b_grid=ew_b_grid,
            seed=ew_b_tune_seed,
            tune_rounds=ew_b_tune_rounds,
        )
        print(
            "[EW-tune] best B={:.6g} from grid {} (tune_rounds={})".format(
                ew_opt_B, [float(v) for v in ew_b_grid], min(ew_b_tune_rounds, n)
            )
        )
        print("[EW-tune] B\tavg_logloss")
        for b, ll in ew_rows:
            print(f"[EW-tune] {b:.6g}\t{ll:.6f}")

    def fmt_b(v):
        return f"{v:.4g}"

    labels = {
        "OGD": "OGD",
        "ONS": "ONS",
        "AIOLI-Bopt": f"AIOLI (B={fmt_b(aioli_opt_B)})",
        "AIOLI-B5": "AIOLI (B=5)",
        "EW-B5": "EW (B=5)",
        "EW-Bopt": f"EW (B={fmt_b(ew_opt_B)})",
    }

    print(f"[data] dataset={meta.name}/{meta.split}, X={X.shape}, repeats={repeats}, B={B}, R={R:.3f}")
    print(f"[data] local_path={meta.local_path}")
    print(f"[mode] unregularized_ogd_ons={unreg_ogd_ons}")
    print(f"[AIOLI] shared={aioli_shared_params}, B*= {aioli_opt_B:.6g}, lam*= {aioli_lam_star:.6g}")
    print(f"[EW] base={ew_params}, B*= {ew_opt_B:.6g}")

    curves_list = []
    for r in range(repeats):
        print(f"[run] repeat {r + 1}/{repeats}")
        curves = run_one_permutation(
            X,
            y,
            B=B,
            R=R,
            ogd_params={**ogd_params, "regularized": not unreg_ogd_ons},
            ons_params={**ons_params, "regularized": not unreg_ogd_ons},
            aioli_opt_B=aioli_opt_B,
            aioli_opt_lam=aioli_lam_star,
            ew_opt_B=ew_opt_B,
            aioli_shared_params=aioli_shared_params,
            ew_params=ew_params,
            seed=seed0 + r,
        )
        curves_list.append(curves)

    mode_tag = "unreg_ogd_ons" if unreg_ogd_ons else "reg_ogd_ons"
    curve_keys = sorted(curves_list[0].keys())

    def safe_curve_key(key):
        out = key
        for old, new in [
            ("-", "_"),
            ("(", ""),
            (")", ""),
            (" ", "_"),
            ("/", "_"),
            ("+", "plus"),
        ]:
            out = out.replace(old, new)
        return f"curve_{out}"

    save_payload = {
        "t": np.arange(1, n + 1, dtype=np.int64),
        "dataset_name": np.asarray([meta.name]),
        "dataset_split": np.asarray([meta.split]),
        "n": np.asarray([n], dtype=np.int64),
        "repeats": np.asarray([repeats], dtype=np.int64),
        "R": np.asarray([R], dtype=np.float64),
        "B_base": np.asarray([B], dtype=np.float64),
        "aioli_opt_B": np.asarray([aioli_opt_B], dtype=np.float64),
        "aioli_lam_star": np.asarray([aioli_lam_star], dtype=np.float64),
        "ew_opt_B": np.asarray([ew_opt_B], dtype=np.float64),
        "aioli_lam_rows": np.asarray(lam_rows, dtype=np.float64),
        "ew_b_rows": np.asarray(ew_rows, dtype=np.float64),
    }
    curve_key_map = {}
    for key in curve_keys:
        safe_key = safe_curve_key(key)
        curve_key_map[safe_key] = key
        save_payload[safe_key] = np.stack([c[key] for c in curves_list], axis=0)

    out_npz = out_dir / f"{meta.name}_{meta.split}_{mode_tag}_online_mala_comparable_results.npz"
    np.savez_compressed(out_npz, **save_payload)

    out_json = out_dir / f"{meta.name}_{meta.split}_{mode_tag}_online_mala_comparable_meta.json"
    out_json.write_text(
        json.dumps(
            {
                "dataset": meta.name,
                "split": meta.split,
                "n": n,
                "repeats": repeats,
                "mode": mode_tag,
                "aioli": {
                    "B_opt": aioli_opt_B,
                    "lambda_opt": aioli_lam_star,
                    "lambda_grid_rows": [[float(a), float(b)] for a, b in lam_rows],
                    "B_fixed": B,
                    "lambda_fixed_for_B5": aioli_b5_lam,
                },
                "ew": {
                    "B_opt": ew_opt_B,
                    "B_grid_rows": [[float(a), float(b)] for a, b in ew_rows],
                    "B_fixed": B,
                    "base_params": ew_params,
                    "tune_rounds": min(ew_b_tune_rounds, n),
                },
                "curve_key_map": curve_key_map,
                "labels": labels,
            },
            indent=2,
        )
    )

    print(f"[saved] {out_npz}")
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()
