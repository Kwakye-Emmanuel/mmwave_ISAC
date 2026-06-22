"""Secrecy rate vs number of sensing beams L.

Professor's experiment:
    X-axis : L = [4, 8, 16, 32, 64, 128]
    Y-axis : Ergodic secrecy sum-rate [bps/Hz]

    Curves:
        Genie-Aided  — uses true g_e  (perfect CSI upper bound)
        Proposed     — uses estimated ĝ_e from MLE sensing
        Gap          — Genie - Proposed

    Fixed  : rho_t = 20 dB
    Phase 1: K = 1 (single user)
    Phase 2: K = 2 (paper setting)

Physical intuition:
    L↑ → more echo snapshots
       → CRB(θ_e) decreases  (tighter angle estimate)
       → θ̂_e → θ_e
       → ĝ_e → g_e
       → Proposed → Genie-Aided
       → Gap → 0

Run:
    python simulate_vs_L.py
    python simulate_vs_L.py --K 2
    python simulate_vs_L.py --K 1 --trials 2000
"""
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from isac.config import SystemConfig
from isac.channel import generate_channels
from isac.sensing import compute_beta_s, run_sensing
from isac.signal import compute_secrecy_sum_rate
from isac.scheduling import oracle_scheduling_genie, mask_to_indices


# ---------------------------------------------------------------------------
# Single trial worker
# ---------------------------------------------------------------------------

def _trial_vs_L(t, L, snr_db, K, cfg, beta_s_mag):
    """One Monte Carlo trial for a given L and K.

    Returns:
        r_genie    : secrecy rate with true g_e   (Genie-Aided)
        r_proposed : secrecy rate with ĝ_e        (Proposed)
        rmse_deg   : angle estimation RMSE [degrees]
    """
    trial_seed = cfg.seed + t + L * 10_000
    rng        = np.random.default_rng(trial_seed)
    P_t        = 10 ** (snr_db / 10.0) * cfg.sigma2_C

    # Channel
    sample = generate_channels(
        M=cfg.M, N=cfg.N, d_0=cfg.d_0, d_be=cfg.d_be,
        eta=cfg.eta, d_cu_min=cfg.d_cu_min, d_cu_max=cfg.d_cu_max,
        sigma_e=cfg.sigma_e,
        theta_E_min=cfg.theta_E_min_rad,
        theta_E_max=cfg.theta_E_max_rad,
        seed=trial_seed,
    )
    H, g_e, theta_E = sample["H"], sample["g_e"], sample["theta_E"]

    # Sensing with L beams
    beta_s_t = beta_s_mag * np.exp(1j * rng.uniform(0, 2 * np.pi))
    state    = run_sensing(
        theta_E, cfg.M, cfg.M_r, L,
        cfg.P_s, cfg.sigma2_s, beta_s_t, seed=trial_seed,
    )
    g_hat_e  = state["beta_hat"] * state["at_hat"]
    rmse_deg = float(np.degrees(abs(state["theta_hat"] - theta_E)))

    # For K=1: schedule the single best user by channel norm
    # For K>1: use oracle with both true and estimated g_e
    if K == 1:
        norms     = np.linalg.norm(H, axis=0)
        best_user = [int(np.argmax(norms))]

        r_genie = compute_secrecy_sum_rate(
            H, g_e, best_user, g_e,
            P_t, cfg.sigma2_e, cfg.sigma2_C, cfg.time_frac, cfg.rho)

        r_proposed = compute_secrecy_sum_rate(
            H, g_e, best_user, g_hat_e,
            P_t, cfg.sigma2_e, cfg.sigma2_C, cfg.time_frac, cfg.rho)

    else:
        # Genie: optimal scheduling with true g_e
        best_mask, r_genie = oracle_scheduling_genie(
            H, g_e, K, P_t,
            cfg.sigma2_e, cfg.sigma2_C, cfg.time_frac, cfg.rho)

        # Proposed: optimal scheduling with estimated g_hat_e for AN
        # (use oracle_scheduling_genie but with g_hat_e for AN design)
        from isac.scheduling import oracle_scheduling_label
        best_mask_p, _ = oracle_scheduling_label(
            H, g_e, K, g_hat_e, P_t,
            cfg.sigma2_e, cfg.sigma2_C, cfg.time_frac, cfg.rho)
        sched_p    = mask_to_indices(best_mask_p)
        r_proposed = compute_secrecy_sum_rate(
            H, g_e, sched_p, g_hat_e,
            P_t, cfg.sigma2_e, cfg.sigma2_C, cfg.time_frac, cfg.rho)

    return r_genie, r_proposed, rmse_deg


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def simulate_rsec_vs_L(
    K:        int   = 1,
    snr_db:   float = 20.0,
    L_values: list  = None,
    n_trials: int   = 2000,
    cfg:      SystemConfig = None,
) -> dict:
    """Simulate ergodic secrecy rate vs number of sensing beams L.

    Args:
        K        : number of scheduled users (1 or 2)
        snr_db   : fixed BS SNR [dB]
        L_values : list of beam counts to sweep
        n_trials : Monte Carlo trials per L
        cfg      : SystemConfig
    Returns:
        results dict with keys:
            L_values, rsec_genie, rsec_proposed, gap, rmse_deg, crb_deg
    """
    if cfg is None:
        cfg = SystemConfig()
    if L_values is None:
        L_values = [4, 8, 16, 32, 64, 128]

    beta_s_mag = compute_beta_s(cfg.d_be, cfg.f_c, cfg.epsilon_dBsm)

    rsec_genie    = np.zeros(len(L_values))
    rsec_proposed = np.zeros(len(L_values))
    rmse_deg_mean = np.zeros(len(L_values))
    crb_deg       = np.zeros(len(L_values))

    print(f"\n  Secrecy Rate vs L  |  K={K}, SNR={snr_db:.0f}dB, "
          f"trials={n_trials}")
    print(f"  {'L':>5} | {'Genie':>10} | {'Proposed':>10} | "
          f"{'Gap':>8} | {'RMSE(deg)':>10} | {'CRB(deg)':>10}")
    print("  " + "-" * 65)

    for i, L in enumerate(L_values):

        results = Parallel(n_jobs=-1)(
            delayed(_trial_vs_L)(t, L, snr_db, K, cfg, beta_s_mag)
            for t in range(n_trials)
        )
        res_np = np.array(results)

        rsec_genie[i]    = np.mean(res_np[:, 0])
        rsec_proposed[i] = np.mean(res_np[:, 1])
        rmse_deg_mean[i] = np.mean(res_np[:, 2])

        # Theoretical CRB at mid-range angle (30 degrees)
        from isac.sensing import crb_theta
        crb_rad    = crb_theta(
            np.radians(30), cfg.M, cfg.M_r, L,
            cfg.P_s, cfg.sigma2_s, abs(beta_s_mag))
        crb_deg[i] = np.degrees(np.sqrt(crb_rad))

        gap = rsec_genie[i] - rsec_proposed[i]
        print(f"  {L:>5} | {rsec_genie[i]:>10.4f} | {rsec_proposed[i]:>10.4f} | "
              f"{gap:>8.4f} | {rmse_deg_mean[i]:>10.4f} | {crb_deg[i]:>10.4f}")

    return {
        "L_values":      L_values,
        "rsec_genie":    rsec_genie,
        "rsec_proposed": rsec_proposed,
        "gap":           rsec_genie - rsec_proposed,
        "rmse_deg":      rmse_deg_mean,
        "crb_deg":       crb_deg,
        "K":             K,
        "snr_db":        snr_db,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_rsec_vs_L(results: dict, save_path: str = None):
    """IEEE-quality plot of secrecy rate and gap vs L."""

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

    L_values = results["L_values"]
    K        = results["K"]
    snr_db   = results["snr_db"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ms, mw = 7, 1.5

    # --- Left: Secrecy rates ---
    ax1.plot(L_values, results["rsec_genie"],
             color="#7f7f7f", linestyle="--", marker="s",
             markerfacecolor="none", markeredgecolor="#7f7f7f",
             markeredgewidth=mw, markersize=ms, label="Genie-Aided")

    ax1.plot(L_values, results["rsec_proposed"],
             color="#0000CD", linestyle="-", marker="o",
             markerfacecolor="none", markeredgecolor="#0000CD",
             markeredgewidth=mw, markersize=ms, label="Proposed")

    ax1.set_xlabel("Number of Sensing Beams $L$", fontsize=11)
    ax1.set_ylabel("Ergodic Secrecy Sum-Rate (bps/Hz)", fontsize=11)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(L_values)
    ax1.set_xticklabels([str(l) for l in L_values])
    ax1.set_ylim(bottom=0)
    ax1.set_title(f"K={K}, $\\rho_t$={snr_db:.0f} dB", fontsize=10)
    ax1.legend(loc="lower right", handlelength=2.5)

    # --- Right: Gap and RMSE ---
    color_gap  = "#CC0000"
    color_rmse = "#228B22"

    ax2.plot(L_values, results["gap"],
             color=color_gap, linestyle="-", marker="^",
             markerfacecolor="none", markeredgecolor=color_gap,
             markeredgewidth=mw, markersize=ms, label="Rate gap (Genie − Proposed)")

    ax2.set_xlabel("Number of Sensing Beams $L$", fontsize=11)
    ax2.set_ylabel("Secrecy Rate Gap (bps/Hz)", fontsize=11, color=color_gap)
    ax2.tick_params(axis="y", labelcolor=color_gap)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(L_values)
    ax2.set_xticklabels([str(l) for l in L_values])
    ax2.set_ylim(bottom=0)
    ax2.set_title(f"Genie−Proposed gap  (K={K})", fontsize=10)

    # RMSE on secondary axis
    ax2b = ax2.twinx()
    ax2b.plot(L_values, results["rmse_deg"],
              color=color_rmse, linestyle="--", marker="s",
              markerfacecolor="none", markeredgecolor=color_rmse,
              markeredgewidth=mw, markersize=ms, label="RMSE (deg)")
    ax2b.plot(L_values, results["crb_deg"],
              color=color_rmse, linestyle=":", marker="",
              label="$\\sqrt{\\mathrm{CRB}}$ (deg)")
    ax2b.set_ylabel("Angle Estimation Error (deg)", fontsize=11,
                    color=color_rmse)
    ax2b.tick_params(axis="y", labelcolor=color_rmse)
    ax2b.set_ylim(bottom=0)

    # Combined legend
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2,
               loc="upper right", fontsize=8, handlelength=2.0)

    plt.tight_layout(pad=0.8)

    if save_path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        ext = save_path.split(".")[-1]
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format=ext)
        print(f"\n  Saved -> {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Secrecy rate vs number of sensing beams L"
    )
    parser.add_argument("--K",       type=int,   default=1,
                        help="Scheduled users (1 or 2)")
    parser.add_argument("--snr",     type=float, default=20.0,
                        help="Fixed BS SNR in dB")
    parser.add_argument("--trials",  type=int,   default=2000,
                        help="Monte Carlo trials per L")
    parser.add_argument("--L",       type=int,   nargs="+",
                        default=[4, 8, 16, 32, 64, 128],
                        help="List of L values to sweep")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    cfg     = SystemConfig()
    cfg.Kd  = args.K

    results = simulate_rsec_vs_L(
        K        = args.K,
        snr_db   = args.snr,
        L_values = args.L,
        n_trials = args.trials,
        cfg      = cfg,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for ext in ["png", "pdf"]:
        fig = plot_rsec_vs_L(
            results,
            save_path=str(cfg.output_dir / f"rsec_vs_L_K{args.K}.{ext}"),
        )
        plt.close(fig)

    # Save raw results
    import os
    np.savez(
        cfg.output_dir / f"rsec_vs_L_K{args.K}.npz",
        **{k: np.array(v) for k, v in results.items()
           if k not in ("K", "snr_db")},
        K      = results["K"],
        snr_db = results["snr_db"],
    )
    print(f"  Raw results saved -> rsec_vs_L_K{args.K}.npz")
    print("\n  Done.")
