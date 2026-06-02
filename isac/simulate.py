"""Monte Carlo simulation for ISAC-aided physical layer security (Parallel Version)."""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from joblib import Parallel, delayed

from .channel import generate_channels
from .config import SystemConfig
from .sensing import compute_beta_s, run_sensing
from .signal import compute_secrecy_sum_rate
from .scheduling import (
    random_scheduling,
    oracle_scheduling_genie,
    mask_to_indices,
)


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def _run_single_trial(t, s_idx, snr, cfg, beta_s_mag):
    trial_seed = cfg.seed + t + s_idx * 100_000
    rng        = np.random.default_rng(trial_seed)
    P_t        = 10 ** (snr / 10.0) * cfg.sigma2_C

    sample = generate_channels(
        M=cfg.M, N=cfg.N, d_0=cfg.d_0, d_be=cfg.d_be,
        eta=cfg.eta, d_cu_min=cfg.d_cu_min, d_cu_max=cfg.d_cu_max,
        sigma_e=cfg.sigma_e,
        theta_E_min=cfg.theta_E_min_rad,
        theta_E_max=cfg.theta_E_max_rad,
        seed=trial_seed,
    )
    H, g_e, theta_E = sample["H"], sample["g_e"], sample["theta_E"]

    beta_s_t = beta_s_mag * np.exp(1j * rng.uniform(0, 2 * np.pi))
    state    = run_sensing(
        theta_E, cfg.M, cfg.M_r, cfg.L,
        cfg.P_s, cfg.sigma2_s, beta_s_t, seed=trial_seed,
    )
    # Coarse Eve CSI estimate: E[|alpha_e|] * a(theta_hat_e)
    # Since alpha_e ~ CN(0, sigma_e^2), we use sigma_e as the
    # expected amplitude. Direction from MLE sensing estimate.
    # Ref: E[|alpha_e|^2] = sigma_e^2 = (d_0/d_be)^eta
    g_hat_e = cfg.sigma_e * state["at_hat"]  # Coarse Eve CSI estimate: expected path loss × estimated steering vector

    # B1 and B2 share the same random scheduling — only AN design differs
    sched_rand = mask_to_indices(random_scheduling(cfg.N, cfg.Kd, rng))

    r_B1 = compute_secrecy_sum_rate(
        H, g_e, sched_rand, None,
        P_t, cfg.sigma2_e, cfg.sigma2_C, cfg.time_frac, cfg.rho)

    r_B2 = compute_secrecy_sum_rate(
        H, g_e, sched_rand, g_hat_e,
        P_t, cfg.sigma2_e, cfg.sigma2_C, cfg.time_frac, cfg.rho)

    _, r_ora = oracle_scheduling_genie(
        H, g_e, cfg.Kd, P_t,
        cfg.sigma2_e, cfg.sigma2_C, cfg.time_frac, cfg.rho)

    return r_B1, r_B2, r_ora


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

def simulate_rsec_vs_snr(cfg: SystemConfig | None = None) -> tuple:
    if cfg is None:
        cfg = SystemConfig()

    beta_s_mag = compute_beta_s(cfg.d_be, cfg.f_c, cfg.epsilon_dBsm)
    snr_db     = np.linspace(cfg.snr_min_dB, cfg.snr_max_dB, cfg.n_snr_pts)

    rsec_B1     = np.zeros(len(snr_db))
    rsec_B2     = np.zeros(len(snr_db))
    rsec_oracle = np.zeros(len(snr_db))
    pout_B1     = np.zeros(len(snr_db))
    pout_B2     = np.zeros(len(snr_db))
    pout_oracle = np.zeros(len(snr_db))

    print(f"  Trials : {cfg.n_trials}  |  SNR pts: {len(snr_db)}")
    print(f"  {'SNR(dB)':>8} | {'B1':>10} | {'B2':>10} | {'Oracle':>10} |")
    print("  " + "-" * 50)

    for s_idx, snr in enumerate(snr_db):
        results = Parallel(n_jobs=-1)(
            delayed(_run_single_trial)(t, s_idx, snr, cfg, beta_s_mag)
            for t in range(cfg.n_trials)
        )
        res_np        = np.array(results)
        r_B1, r_B2, r_ora = res_np[:, 0], res_np[:, 1], res_np[:, 2]

        rsec_B1[s_idx]     = np.mean(r_B1)
        rsec_B2[s_idx]     = np.mean(r_B2)
        rsec_oracle[s_idx] = np.mean(r_ora)
        pout_B1[s_idx]     = np.mean(r_B1  < cfg.R0)
        pout_B2[s_idx]     = np.mean(r_B2  < cfg.R0)
        pout_oracle[s_idx] = np.mean(r_ora < cfg.R0)

        print(f"  {snr:>+8.1f} | {rsec_B1[s_idx]:>10.4f} | "
              f"{rsec_B2[s_idx]:>10.4f} | {rsec_oracle[s_idx]:>10.4f} |")

    return snr_db, rsec_B1, rsec_B2, rsec_oracle, pout_B1, pout_B2, pout_oracle


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------

def _apply_rcparams():
    plt.rcParams.update({
        "font.family":         "serif",
        "font.serif":          ["Times New Roman", "DejaVu Serif"],
        "font.size":           11,
        "axes.linewidth":      0.8,
        "axes.grid":           True,
        "grid.linestyle":      "--",
        "grid.alpha":          0.35,
        "grid.linewidth":      0.5,
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


def _topology_tag(cfg):
    return (f"_Eve{cfg.eve_snr_dB:.0f}dB"
            f"_dbe{cfg.d_be:.0f}m"
            f"_CU{cfg.d_cu_min:.0f}-{cfg.d_cu_max:.0f}m")


# ---------------------------------------------------------------------------
# Secrecy sum-rate plot
# ---------------------------------------------------------------------------

def plot_rsec_vs_snr(
    snr_db, rsec_B1, rsec_B2, rsec_oracle,
    rsec_proposed=None, rsec_deepsets=None, rsec_conv_dl=None,
    cfg=None, save_path=None,
):
    if cfg is None:
        cfg = SystemConfig()
    _apply_rcparams()

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ms, mw  = 7, 1.5

    ax.plot(snr_db, rsec_oracle,
            color="#7f7f7f", linestyle="--", marker="s",
            markerfacecolor="none", markeredgecolor="#7f7f7f",
            markeredgewidth=mw, markersize=ms, label="Genie-Aided")

    if rsec_proposed is not None:
        ax.plot(snr_db, rsec_proposed,
                color="#0000CD", linestyle="-", marker="s",
                markerfacecolor="none", markeredgecolor="#0000CD",
                markeredgewidth=mw, markersize=ms, label="Proposed")

    if rsec_deepsets is not None:
        ax.plot(snr_db, rsec_deepsets,
                color="#CC0000", linestyle="-", marker="o",
                markerfacecolor="none", markeredgecolor="#CC0000",
                markeredgewidth=mw, markersize=ms, label="DeepSets")

    if rsec_conv_dl is not None:
        ax.plot(snr_db, rsec_conv_dl,
                color="#228B22", linestyle="-", marker="D",
                markerfacecolor="none", markeredgecolor="#228B22",
                markeredgewidth=mw, markersize=ms,
                label="DL Sched. (Isotropic AN)")

    ax.plot(snr_db, rsec_B2,
            color="black", linestyle="--", marker="s",
            markerfacecolor="none", markeredgewidth=mw, markersize=ms,
            label="RS + Directed AN (B2)")

    ax.plot(snr_db, rsec_B1,
            color="black", linestyle="-", marker="o",
            markerfacecolor="none", markeredgewidth=mw, markersize=ms,
            label="RS + Isotropic AN (B1)")

    ax.set_xlabel("SNR, $\\rho_t$ (dB)", fontsize=11)
    ax.set_ylabel("Ergodic Secrecy Sum-Rate (bps/Hz)", fontsize=11)
    ax.set_xlim(left=snr_db[0], right=snr_db[-1])
    ax.set_ylim(bottom=0)
    ax.margins(0)
    ax.set_xticks(np.arange(snr_db[0], snr_db[-1] + 1, 2))
    ax.legend(loc="best", handlelength=2.5, borderpad=0.6, labelspacing=0.4)
    plt.tight_layout(pad=0.5)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        if "rsec_vs_snr" in save_path:
            base, ext_str = save_path.rsplit(".", 1)
            save_path = f"{base}{_topology_tag(cfg)}.{ext_str}"
        ext = save_path.split(".")[-1]
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format=ext)
        print(f"\n  Saved -> {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Secrecy outage probability plot
# ---------------------------------------------------------------------------

def plot_outage_vs_snr(
    snr_db, pout_B1, pout_B2, pout_oracle,
    pout_proposed=None, pout_conv_dl=None,
    cfg=None, save_path=None,
):
    if cfg is None:
        cfg = SystemConfig()
    _apply_rcparams()

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ms, mw  = 7, 1.5

    def mask(arr):
        out = arr.copy().astype(float)
        out[out == 0.0] = np.nan
        return out

    ax.semilogy(snr_db, mask(pout_oracle),
                color="#7f7f7f", linestyle="--", marker="s",
                markerfacecolor="none", markeredgecolor="#7f7f7f",
                markeredgewidth=mw, markersize=ms, label="Genie-Aided")

    if pout_proposed is not None:
        ax.semilogy(snr_db, mask(pout_proposed),
                    color="#0000CD", linestyle="-", marker="s",
                    markerfacecolor="none", markeredgecolor="#0000CD",
                    markeredgewidth=mw, markersize=ms, label="Proposed")

    if pout_conv_dl is not None:
        ax.semilogy(snr_db, mask(pout_conv_dl),
                    color="#228B22", linestyle="-", marker="D",
                    markerfacecolor="none", markeredgecolor="#228B22",
                    markeredgewidth=mw, markersize=ms,
                    label="DL Sched. (Isotropic AN)")

    ax.semilogy(snr_db, mask(pout_B2),
                color="black", linestyle="--", marker="s",
                markerfacecolor="none", markeredgewidth=mw, markersize=ms,
                label="RS + Directed AN (B2)")

    ax.semilogy(snr_db, mask(pout_B1),
                color="black", linestyle="-", marker="o",
                markerfacecolor="none", markeredgewidth=mw, markersize=ms,
                label="RS + Isotropic AN (B1)")

    ax.set_xlabel("SNR, $\\rho_t$ (dB)", fontsize=11)
    ax.set_ylabel("Secrecy Outage Probability", fontsize=11)
    ax.set_xlim(left=snr_db[0], right=snr_db[-1])
    ax.set_ylim(bottom=1e-4, top=1.5)
    ax.margins(x=0)
    ax.set_xticks(np.arange(snr_db[0], snr_db[-1] + 1, 2))
    ax.legend(loc="best", handlelength=2.5, borderpad=0.6, labelspacing=0.4)
    plt.tight_layout(pad=0.5)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        if "outage_vs_snr" in save_path:
            base, ext_str = save_path.rsplit(".", 1)
            save_path = f"{base}{_topology_tag(cfg)}.{ext_str}"
        ext = save_path.split(".")[-1]
        fig.savefig(save_path, dpi=300, bbox_inches="tight", format=ext)
        print(f"\n  Saved -> {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    cfg = SystemConfig()
    print(cfg.summary())
    print()

    snr_db, rsec_B1, rsec_B2, rsec_oracle, pout_B1, pout_B2, pout_oracle = \
        simulate_rsec_vs_snr(cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for ext in ["png", "pdf"]:
        fig = plot_rsec_vs_snr(
            snr_db, rsec_B1, rsec_B2, rsec_oracle,
            cfg=cfg,
            save_path=str(cfg.output_dir / f"sim_rsec_vs_snr.{ext}"),
        )
        plt.close(fig)

    for ext in ["png", "pdf"]:
        fig = plot_outage_vs_snr(
            snr_db, pout_B1, pout_B2, pout_oracle,
            cfg=cfg,
            save_path=str(cfg.output_dir / f"sim_outage_vs_snr.{ext}"),
        )
        plt.close(fig)

    print("\n  Done.")
