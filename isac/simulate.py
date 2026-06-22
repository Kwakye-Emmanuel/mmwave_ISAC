"""simulate.py — Secrecy sum-rate vs P0 sweep.

Evaluates 4 scheduling/beamforming schemes across P0 range.
Saves plot to outputs/ and raw results to outputs/results.npz.

Schemes:
    1. Proposed       — GFlowNet + directed AN
    2. GFlownet iso AN     — GFlowNet + isotropic AN
    3. RS dir AN      — random scheduling + directed AN
    4. RS iso AN      — random scheduling + isotropic AN
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import torch

from config import SystemConfig, GFlowNetConfig
from channel_mmwave import generate_channels_mmwave
from gflownet import GFlowNet, build_features, compute_reward
from sensing import run_sensing


# ----------------------------------------------------------------
# IEEE plot style
# ----------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.grid":          True,
    "grid.color":         "#cccccc",
    "grid.linewidth":     0.6,
    "grid.linestyle":     "--",
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.linewidth":     0.8,
    "font.family":        "serif",
    "font.size":          12,
    "axes.labelsize":     13,
    "legend.fontsize":    11,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "lines.linewidth":    1.8,
    "lines.markersize":   7,
})


def run_simulation(
    sys_cfg  : SystemConfig,
    gfn_cfg  : GFlowNetConfig,
    gfn      : GFlowNet,
    rng      : np.random.Generator,
) -> dict:
    """Sweep P0 and evaluate all 4 schemes.

    Returns dict of results arrays shape (n_P0_pts,).
    """
    n_pts = sys_cfg.n_P0_pts

    results = {
        "gfn_directed"   : np.zeros(n_pts),
        "gfn_iso"    : np.zeros(n_pts),
        "rs_dir"     : np.zeros(n_pts),
        "rs_iso"     : np.zeros(n_pts),
        "P0_dBm"     : sys_cfg.P0_dBm_range,
    }

    for idx, (P0_dBm, P0) in enumerate(
        zip(sys_cfg.P0_dBm_range, sys_cfg.P0_range)
    ):
        print(f"\n[P0 = {P0_dBm:.2f} dBm = {P0:.4f} W]")

        gfn_dir_rates  =  []
        gfn_iso_rates  = []
        rs_dir_rates   = []
        rs_iso_rates   = []

        for trial in range(sys_cfg.n_trials):

            # 1. channel realization
            ch = generate_channels_mmwave(
                N_t             = sys_cfg.N_t,
                N               = sys_cfg.N,
                kappa           = sys_cfg.kappa,
                L_p             = sys_cfg.L_p,
                sigma_alpha     = sys_cfg.sigma_alpha,
                theta_E_min_deg = sys_cfg.theta_E_min_deg,
                theta_E_max_deg = sys_cfg.theta_E_max_deg,
                seed            = int(rng.integers(0, 2**31)),
            )
            H       = ch["H"]

            # 2. sensing — uses same P0 budget
            beta = sys_cfg.beta_mag * np.exp(
                1j * rng.uniform(0, 2 * np.pi)
            )
            sensing_state = run_sensing(
                theta_E_deg = ch["theta_E"],
                beta        = beta,
                N_t         = sys_cfg.N_t,
                N_r         = sys_cfg.N_r,
                L           = sys_cfg.L,
                P0          = P0,
                sigma2_R    = sys_cfg.sigma2_R,
                seed        = int(rng.integers(0, 2**31)),
            )

            theta_hat = sensing_state["theta_hat"]
            crb       = sensing_state["crb"]
            g_e_hat   = sensing_state["g_e_hat"]

            # build features for GFlowNet
            user_feats_np, eve_ctx_np = build_features(
                H, sensing_state, sys_cfg.N
            )

            # common reward kwargs
            reward_kwargs = dict(
                H         = H,
                theta_hat = theta_hat,
                crb       = crb,
                P_t       = P0,
                sys_cfg   = sys_cfg,
                rng       = rng,
                n_mc      = gfn_cfg.n_mc,
            )

            # ── Scheme 1: Proposed — GFlowNet + directed AN ──
            selected_gfn = gfn.sample(
                user_feats_np, eve_ctx_np, greedy=True
            )
            r1 = compute_reward(
                selected_gfn,
                an_type = "directed",
                g_e_hat = g_e_hat,
                **reward_kwargs,
            )
            gfn_dir_rates.append(r1)

            # ── Scheme 2: DLS iso AN — GFlowNet + isotropic AN ──
            r2 = compute_reward(
                selected_gfn,
                an_type = "isotropic",
                **reward_kwargs,
            )
            gfn_iso_rates.append(r2)

            # ── Scheme 3: RS dir AN — random + directed AN ──
            selected_rand = tuple(
                rng.choice(sys_cfg.N, sys_cfg.K, replace=False)
            )
            r3 = compute_reward(
                selected_rand,
                an_type = "directed",
                g_e_hat = g_e_hat,
                **reward_kwargs,
            )
            rs_dir_rates.append(r3)

            # ── Scheme 4: RS iso AN — random + isotropic AN ──
            r4 = compute_reward(
                selected_rand,
                an_type = "isotropic",
                **reward_kwargs,
            )
            rs_iso_rates.append(r4)

            if (trial + 1) % 100 == 0:
                print(f"  trial {trial+1}/{sys_cfg.n_trials}  "
                      f"proposed={np.mean(gfn_dir_rates):.3f}  "
                      f"rs_iso={np.mean(rs_iso_rates):.3f}")

        results["gfn_directed"][idx] = np.mean(gfn_dir_rates)
        results["gfn_iso"][idx]  = np.mean(gfn_iso_rates)
        results["rs_dir"][idx]   = np.mean(rs_dir_rates)
        results["rs_iso"][idx]   = np.mean(rs_iso_rates)

        print(f"  → GFN_dir={results['gfn_directed'][idx]:.4f}  "
              f"GFN_iso={results['gfn_iso'][idx]:.4f}  "
              f"RS_dir={results['rs_dir'][idx]:.4f}  "
              f"RS_iso={results['rs_iso'][idx]:.4f}")

    return results


def plot_results(results: dict, out_dir: Path) -> None:
    """Plot secrecy sum-rate vs P0 — IEEE style, no title."""

    P0_dBm = results["P0_dBm"]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(P0_dBm, results["gfn_directed"],
            "o-", color="#0055A4",
            markerfacecolor="white", markeredgewidth=1.8,
            label="(GFlowNet with directed AN")

    ax.plot(P0_dBm, results["gfn_iso"],
            "s-", color="#009E73",
            markerfacecolor="white", markeredgewidth=1.8,
            label="GFlowNet with isotropic AN")

    ax.plot(P0_dBm, results["rs_dir"],
            "^--", color="#000000",
            markerfacecolor="white", markeredgewidth=1.8,
            label="RS with directed AN")

    ax.plot(P0_dBm, results["rs_iso"],
            "D-", color="#444444",
            markerfacecolor="white", markeredgewidth=1.8,
            label="RS with isotropic AN")

    ax.set_xlabel(r"$P_0$ (dBm)")
    ax.set_ylabel("Secrecy Sum-Rate (bits/s/Hz)")
    ax.set_xlim([P0_dBm[0], P0_dBm[-1]])
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2.5))
    ax.legend(frameon=True, edgecolor="#cccccc",
              fancybox=False, loc="upper left")

    fig.tight_layout()

    out_path = out_dir / "secrecy_rate_vs_P0.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved → {out_path}")


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
if __name__ == "__main__":

    sys_cfg = SystemConfig()
    gfn_cfg = GFlowNetConfig(N=sys_cfg.N, K=sys_cfg.K)
    rng     = np.random.default_rng(sys_cfg.seed + 1)   # different from training

    # ── Train GFlowNet ──────────────────────────────────────────
    print("=" * 60)
    print("  Phase 1: Training GFlowNet")
    print("=" * 60)

    gfn = GFlowNet(gfn_cfg)
    rng_train = np.random.default_rng(sys_cfg.seed)

    # train at middle of P0 sweep
    P_t_train = 10**(30.0/10.0) * 1e-3   # 30 dBm = 1.0 W

    gfn.train(
        sys_cfg   = sys_cfg,
        P_t       = P_t_train,
        rng       = rng_train,
        log_every = gfn_cfg.log_every,
    )
    print(f"Training complete. lnZ = {gfn.log_z.item():.4f}")

    # ── Run simulation ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase 1: P0 sweep evaluation")
    print(f"  P0 range: {sys_cfg.P0_dBm_min} - {sys_cfg.P0_dBm_max} dBm")
    print(f"  Trials per point: {sys_cfg.n_trials}")
    print(f"  Schemes: GFlowNet directed, GFlowNet iso, RS directed, RS iso")
    print("=" * 60)

    results = run_simulation(sys_cfg, gfn_cfg, gfn, rng)

    # ── Save raw results ─────────────────────────────────────────
    out_dir = sys_cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "results.npz",
        **results
    )
    print(f"Raw results saved → {out_dir / 'results.npz'}")

    # ── Plot ─────────────────────────────────────────────────────
    plot_results(results, out_dir)