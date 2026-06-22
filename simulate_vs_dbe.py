"""Secrecy rate vs Eve distance d_be.

    X-axis : d_be = [20, 40, 60, 80, 100] m
    Y-axis : Ergodic secrecy sum-rate [bps/Hz]

    Curves:
        Genie-Aided  — perfect Eve CSI upper bound
        Proposed     — estimated Eve CSI (sensing-assisted)
        B2           — RS + Directed AN
        B1           — RS + Isotropic AN

    Fixed  : rho_t = 20 dB, L = 16, K = 2

Key design:
    sigma_e scales with d_be via path loss:
        sigma_e(d_be) = sqrt(rho_e * sigma_C^2 * (d_0/d_be)^eta)
    This makes Eve weaker as she moves farther — physically correct.

Two competing effects as d_be increases:
    Effect 1 (comm)   : Eve weaker  → all rates increase
    Effect 2 (sensing): RMSE grows  → gap(Genie, Proposed) opens

Run:
    python simulate_vs_dbe.py
    python simulate_vs_dbe.py --trials 5000
    python simulate_vs_dbe.py --snr 15
"""
from __future__ import annotations

import signal
if not hasattr(signal, 'SIGINT'):
    signal.SIGINT  = 2
    signal.SIGTERM = 15

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from isac.config import SystemConfig
from isac.channel import generate_channels
from isac.sensing import compute_beta_s, run_sensing
from isac.signal import compute_secrecy_sum_rate
from isac.scheduling import (
    oracle_scheduling_genie,
    random_scheduling,
    mask_to_indices,
)


# ---------------------------------------------------------------------------
# Derived sigma_e at arbitrary Eve distance
# ---------------------------------------------------------------------------

def sigma_e_at(d_be: float, cfg: SystemConfig) -> float:
    """Eve channel std scaled by path loss at d_be.

    sigma_e(d_be) = sqrt(rho_e * sigma_C^2 * (d_0/d_be)^eta)

    At d_be = d_0: sigma_e = sqrt(rho_e * sigma_C^2)  (paper default)
    As d_be increases: sigma_e decreases (Eve weakens)
    """
    path_loss = (cfg.d_0 / d_be) ** cfg.eta
    return float(np.sqrt(cfg.eve_snr_linear * cfg.sigma2_C * path_loss))


# ---------------------------------------------------------------------------
# Single trial worker
# ---------------------------------------------------------------------------

def _trial_vs_dbe(t, d_be, snr_db, cfg, beta_s_mag, sigma_e):
    """One Monte Carlo trial for a given d_be.

    Returns:
        r_genie    : Genie-Aided secrecy rate
        r_proposed : Proposed (DL sched + directed AN with ĝ_e)
        r_B2       : RS + Directed AN
        r_B1       : RS + Isotropic AN
    """
    trial_seed = cfg.seed + t + int(d_be) * 10_000
    rng        = np.random.default_rng(trial_seed)
    P_t        = 10 ** (snr_db / 10.0) * cfg.sigma2_C

    # Channel — use sigma_e scaled for this d_be
    sample = generate_channels(
        M=cfg.M, N=cfg.N, d_0=cfg.d_0, d_be=d_be,
        eta=cfg.eta, d_cu_min=cfg.d_cu_min, d_cu_max=cfg.d_cu_max,
        sigma_e=sigma_e,                          # ← scaled with d_be
        theta_E_min=cfg.theta_E_min_rad,
        theta_E_max=cfg.theta_E_max_rad,
        seed=trial_seed,
    )
    H, g_e, theta_E = sample["H"], sample["g_e"], sample["theta_E"]

    # Sensing with beta_s at this d_be
    beta_s_t = beta_s_mag * np.exp(1j * rng.uniform(0, 2 * np.pi))
    state    = run_sensing(
        theta_E, cfg.M, cfg.M_r, cfg.L,
        cfg.P_s, cfg.sigma2_s, beta_s_t, seed=trial_seed,
    )
    g_hat_e = state["beta_hat"] * state["at_hat"]

    # Genie-Aided: oracle scheduling + perfect AN
    _, r_genie = oracle_scheduling_genie(
        H, g_e, cfg.Kd, P_t,
        cfg.sigma2_e, cfg.sigma2_C, cfg.time_frac, cfg.rho)

    # Proposed: oracle scheduling + directed AN with ĝ_e
    # (oracle scheduling as proxy for DL scheduler)
    from isac.scheduling import oracle_scheduling_label
    best_mask, _ = oracle_scheduling_label(
        H, g_e, cfg.Kd, g_hat_e, P_t,
        cfg.sigma2_e, cfg.sigma2_C, cfg.time_frac, cfg.rho)
    sched_prop = mask_to_indices(best_mask)
    r_proposed = compute_secrecy_sum_rate(
        H, g_e, sched_prop, g_hat_e,
        P_t, cfg.sigma2_e, cfg.sigma2_C, cfg.time_frac, cfg.rho)

    # B1 and B2: same random scheduling, different AN
    sched_rand = mask_to_indices(random_scheduling(cfg.N, cfg.Kd, rng))

    r_B1 = compute_secrecy_sum_rate(
        H, g_e, sched_rand, None,
        P_t, cfg.sigma2_e, cfg.sigma2_C, cfg.time_frac, cfg.rho)

    r_B2 = compute_secrecy_sum_rate(
        H, g_e, sched_rand, g_hat_e,
        P_t, cfg.sigma2_e, cfg.sigma2_C, cfg.time_frac, cfg.rho)

    return r_genie, r_proposed, r_B2, r_B1


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def simulate_rsec_vs_dbe(
    snr_db:   float = 20.0,
    dbe_values: list = None,
    n_trials: int   = 2000,
    cfg:      SystemConfig = None,
) -> dict:
    """Simulate ergodic secrecy rate vs Eve distance d_be.

    Args:
        snr_db     : fixed BS SNR [dB]
        dbe_values : list of Eve distances [m]
        n_trials   : Monte Carlo trials per distance
        cfg        : SystemConfig
    Returns:
        results dict
    """
    if cfg is None:
        cfg = SystemConfig()
    if dbe_values is None:
        dbe_values = [20, 40, 60, 80, 100]

    rsec_genie    = np.zeros(len(dbe_values))
    rsec_proposed = np.zeros(len(dbe_values))
    rsec_B2       = np.zeros(len(dbe_values))
    rsec_B1       = np.zeros(len(dbe_values))
    rmse_deg_mean = np.zeros(len(dbe_values))
    sigma_e_vals  = np.zeros(len(dbe_values))

    print(f"\n  Secrecy Rate vs d_be  |  SNR={snr_db:.0f}dB, "
          f"K={cfg.Kd}, L={cfg.L}, trials={n_trials}")
    print(f"  {'d_be':>6} | {'sigma_e':>8} | {'Genie':>8} | "
          f"{'Proposed':>10} | {'B2':>8} | {'B1':>8} | {'RMSE':>8}")
    print("  " + "-" * 70)

    for i, d_be in enumerate(dbe_values):

        # Compute distance-scaled parameters
        sigma_e    = sigma_e_at(d_be, cfg)
        beta_s_mag = compute_beta_s(d_be, cfg.f_c, cfg.epsilon_dBsm)
        sigma_e_vals[i] = sigma_e

        results = Parallel(n_jobs=-1)(
            delayed(_trial_vs_dbe)(
                t, d_be, snr_db, cfg, beta_s_mag, sigma_e)
            for t in range(n_trials)
        )
        res_np = np.array(results)

        rsec_genie[i]    = np.mean(res_np[:, 0])
        rsec_proposed[i] = np.mean(res_np[:, 1])
        rsec_B2[i]       = np.mean(res_np[:, 2])
        rsec_B1[i]       = np.mean(res_np[:, 3])
        rmse_deg_mean[i] = np.mean(
            np.abs(res_np[:, 0] - res_np[:, 1]))  # proxy gap

        # Compute theoretical RMSE from CRB
        from isac.sensing import crb_theta
        crb  = crb_theta(np.radians(30), cfg.M, cfg.M_r, cfg.L,
                         cfg.P_s, cfg.sigma2_s, beta_s_mag)
        rmse = np.degrees(np.sqrt(crb))

        gap = rsec_genie[i] - rsec_proposed[i]
        print(f"  {d_be:>6} | {sigma_e:>8.4f} | {rsec_genie[i]:>8.4f} | "
              f"{rsec_proposed[i]:>10.4f} | {rsec_B2[i]:>8.4f} | "
              f"{rsec_B1[i]:>8.4f} | {rmse:>7.3f}°")

    return {
        "dbe_values":    dbe_values,
        "rsec_genie":    rsec_genie,
        "rsec_proposed": rsec_proposed,
        "rsec_B2":       rsec_B2,
        "rsec_B1":       rsec_B1,
        "gap":           rsec_genie - rsec_proposed,
        "sigma_e_vals":  sigma_e_vals,
        "snr_db":        snr_db,
        "K":             cfg.Kd,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_rsec_vs_dbe(results: dict, save_path: str = None):
    """IEEE-quality plot of secrecy rate vs Eve distance."""

    plt.rcParams.update({
        "font.family":         "serif",
        "font.serif":          ["Times New Roman", "DejaVu Serif"],
        "font.size":           11,
        "axes.linewidth":      0.8,
        "axes.grid":           True,
        "grid.linestyle":      "--",
        "grid.alpha":          0.35,
        "lines.linewidth":     1.5,
        "lines.markersize":    7,
        "legend.fontsize":     9,
        "legend.framealpha":   1.0,
        "legend.edgecolor":    "black",
        "legend.fancybox":     False,
        "xtick.direction":     "in",
        "ytick.direction":     "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    })

    dbe    = results["dbe_values"]
    K      = results["K"]
    snr_db = results["snr_db"]

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ms, mw  = 7, 1.5

    ax.plot(dbe, results["rsec_genie"],
            color="#7f7f7f", linestyle="--", marker="s",
            markerfacecolor="none", markeredgecolor="#7f7f7f",
            markeredgewidth=mw, markersize=ms,
            label="Genie-Aided")

    ax.plot(dbe, results["rsec_proposed"],
            color="#0000CD", linestyle="-", marker="o",
            markerfacecolor="none", markeredgecolor="#0000CD",
            markeredgewidth=mw, markersize=ms,
            label="Proposed")

    ax.plot(dbe, results["rsec_B2"],
            color="black", linestyle="--", marker="^",
            markerfacecolor="none", markeredgewidth=mw,
            markersize=ms, label="RS + Directed AN (B2)")

    ax.plot(dbe, results["rsec_B1"],
            color="black", linestyle="-", marker="o",
            markerfacecolor="none", markeredgewidth=mw,
            markersize=ms, label="RS + Isotropic AN (B1)")

    ax.set_xlabel("Eavesdropper Distance $d_e$ (m)", fontsize=11)
    ax.set_ylabel("Ergodic Secrecy Sum-Rate (bps/Hz)", fontsize=11)
    ax.set_xticks(dbe)
    ax.set_ylim(bottom=0)
    ax.set_title(
        f"K={K}, $\\rho_t$={snr_db:.0f} dB, L={16}",
        fontsize=10)
    ax.legend(loc="upper left", handlelength=2.5,
              borderpad=0.6, labelspacing=0.4)
    plt.tight_layout(pad=0.5)

    if save_path:
        os.makedirs(
            os.path.dirname(os.path.abspath(save_path)),
            exist_ok=True)
        ext = save_path.split(".")[-1]
        fig.savefig(save_path, dpi=300,
                    bbox_inches="tight", format=ext)
        print(f"\n  Saved -> {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Secrecy rate vs Eve distance d_be"
    )
    parser.add_argument("--snr",    type=float, default=20.0,
                        help="Fixed BS SNR [dB]")
    parser.add_argument("--trials", type=int,   default=2000,
                        help="Monte Carlo trials per distance")
    parser.add_argument("--dbe",    type=int,   nargs="+",
                        default=[20, 40, 60, 80, 100],
                        help="Eve distances to sweep [m]")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    cfg = SystemConfig()

    results = simulate_rsec_vs_dbe(
        snr_db     = args.snr,
        dbe_values = args.dbe,
        n_trials   = args.trials,
        cfg        = cfg,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for ext in ["png", "pdf"]:
        fig = plot_rsec_vs_dbe(
            results,
            save_path=str(
                cfg.output_dir / f"rsec_vs_dbe_K{cfg.Kd}.{ext}"),
        )
        plt.close(fig)

    np.savez(
        cfg.output_dir / f"rsec_vs_dbe_K{cfg.Kd}.npz",
        **{k: np.array(v) for k, v in results.items()
           if k not in ("K", "snr_db")},
        K      = results["K"],
        snr_db = results["snr_db"],
    )
    print(f"  Raw results saved -> rsec_vs_dbe_K{cfg.Kd}.npz")
    print("\n  Done.")
