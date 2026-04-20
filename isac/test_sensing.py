"""Sensing stage validation tests.

Test 1 (Q1): Empirical MLE RMSE vs theoretical sqrt(CRB) across sensing SNR.
Test 2 (Q3): Theoretical CRB vs number of sensing beams L.

Run:
    python test_sensing.py

Reference:
    [Cao25] Y. Cao et al., IEEE TWC, 2025.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from sensing import (
    ula_steering,
    dft_codebook,
    compute_beta_s,
    simulate_echo,
    mle_estimate,
    crb_theta,
)

# ---------------------------------------------------------------------------
# Parameters (Cao et al. TWC 2025 settings)
# ---------------------------------------------------------------------------

CFG = dict(
    M_t          = 8,        # transmit antennas
    M_r          = 16,       # receive antennas
    L            = 32,       # sensing beams (default)
    theta_E_deg  = 30.0,     # true Eve angle [degrees]
    P_s_dBm      = 20.0,     # sensing transmit power [dBm]
    sigma2_dBm   = -110.0,   # noise power [dBm]
    d_be         = 100.0,    # BS-Eve distance [m]
    f_c          = 28e9,     # carrier frequency [Hz]
    epsilon_dBsm = 7.0,      # radar cross section [dBsm]
    n_trials     = 200,      # Monte Carlo trials per SNR point
    seed         = 42,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

PLOT_STYLE = {
    "font.family":     "serif",
    "font.size":       11,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "lines.linewidth": 1.8,
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _to_watts(dBm: float) -> float:
    return 10 ** ((dBm - 30.0) / 10.0)


def _sigma2_from_snr(snr_dB: float, P_s: float, beta_mag: float, M_t: int) -> float:
    """Derive noise variance from sensing SNR definition:
        SNR_s = P_s * |beta_s|^2 * M_t / sigma2
    """
    return P_s * beta_mag ** 2 * M_t / (10 ** (snr_dB / 10.0))


# ---------------------------------------------------------------------------
# Test 1: RMSE vs Sensing SNR  (Q1)
# ---------------------------------------------------------------------------

def test_rmse_vs_snr() -> tuple[NDArray, NDArray, NDArray]:
    """Compute empirical MLE RMSE and theoretical sqrt(CRB) vs sensing SNR.

    Returns:
        snr_db    : (n_pts,) SNR x-axis [dB]
        rmse_deg  : (n_pts,) empirical RMSE [degrees]
        crb_deg   : (n_pts,) theoretical sqrt(CRB) [degrees]
    """
    M_t   = CFG["M_t"]
    M_r   = CFG["M_r"]
    L     = CFG["L"]
    theta_E  = np.deg2rad(CFG["theta_E_deg"])
    P_s      = _to_watts(CFG["P_s_dBm"])
    beta_mag = compute_beta_s(CFG["d_be"], CFG["f_c"], CFG["epsilon_dBsm"])
    n_trials = CFG["n_trials"]
    seed     = CFG["seed"]

    snr_db   = np.linspace(-10, 20, 8)
    rmse_deg = np.zeros(len(snr_db))
    crb_deg  = np.zeros(len(snr_db))

    theta_grid = np.linspace(-np.pi / 2, np.pi / 2, 1801)

    print("  Test 1: RMSE vs Sensing SNR")
    print(f"  {'SNR (dB)':>10} | {'RMSE (deg)':>12} | {'sqrt(CRB) (deg)':>16}")
    print("  " + "-" * 45)

    for i, snr in enumerate(snr_db):
        sigma2_s = _sigma2_from_snr(snr, P_s, beta_mag, M_t)
        crb      = crb_theta(theta_E, M_t, M_r, L, P_s, sigma2_s, beta_mag)
        crb_deg[i] = np.sqrt(crb) * 180.0 / np.pi

        errors = np.zeros(n_trials)
        for t in range(n_trials):
            rng      = np.random.default_rng(seed + t + i * 10_000)
            beta_s_t = beta_mag * np.exp(1j * rng.uniform(0, 2 * np.pi))
            Y        = simulate_echo(
                theta_E, M_t, M_r, L, P_s, sigma2_s, beta_s_t,
                seed=seed + t + i * 10_000,
            )
            theta_hat, _ = mle_estimate(Y, M_t, M_r, L, P_s, theta_grid)
            errors[t] = (theta_hat - theta_E) * 180.0 / np.pi

        rmse_deg[i] = np.sqrt(np.mean(errors ** 2))
        print(f"  {snr:>+10.1f} | {rmse_deg[i]:>12.4f} | {crb_deg[i]:>16.4f}")

    return snr_db, rmse_deg, crb_deg


# ---------------------------------------------------------------------------
# Test 2: CRB vs L  (Q3, analytical only)
# ---------------------------------------------------------------------------

def test_crb_vs_L() -> tuple[NDArray, NDArray]:
    """Compute theoretical sqrt(CRB) vs number of sensing beams L.

    Returns:
        L_vals  : (n_pts,) beam counts
        crb_deg : (n_pts,) sqrt(CRB) [degrees]
    """
    M_t      = CFG["M_t"]
    M_r      = CFG["M_r"]
    theta_E  = np.deg2rad(CFG["theta_E_deg"])
    P_s      = _to_watts(CFG["P_s_dBm"])
    sigma2_s = _to_watts(CFG["sigma2_dBm"])
    beta_mag = compute_beta_s(CFG["d_be"], CFG["f_c"], CFG["epsilon_dBsm"])

    L_vals  = np.arange(M_t, 65, 4)           # L = 8, 12, 16, ..., 64
    crb_deg = np.zeros(len(L_vals))

    print("\n  Test 2: CRB vs L (analytical)")
    print(f"  {'L':>6} | {'sqrt(CRB) (deg)':>16} | {'3*sqrt(CRB) (deg)':>20}")
    print("  " + "-" * 48)

    for i, L in enumerate(L_vals):
        crb = crb_theta(theta_E, M_t, M_r, int(L), P_s, sigma2_s, beta_mag)
        crb_deg[i] = np.sqrt(crb) * 180.0 / np.pi
        print(f"  {L:>6} | {crb_deg[i]:>16.4f} | {3*crb_deg[i]:>20.4f}")

    return L_vals, crb_deg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rmse_vs_snr(
    snr_db:   NDArray,
    rmse_deg: NDArray,
    crb_deg:  NDArray,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot MLE RMSE vs sensing SNR with sqrt(CRB) lower bound."""
    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(7, 4.8))

    ax.semilogy(snr_db, rmse_deg, "b-o", ms=6,
                label="Empirical MLE RMSE")
    ax.semilogy(snr_db, crb_deg,  "r--s", ms=5,
                label=r"Theoretical $\sqrt{\mathrm{CRB}}$ (lower bound)")

    ax.set_xlabel("Sensing SNR [dB]", fontsize=11)
    ax.set_ylabel("Angle estimation error [degrees]", fontsize=11)
    ax.set_title(
        f"MLE RMSE vs Sensing SNR\n"
        f"($M_t$={CFG['M_t']}, $M_r$={CFG['M_r']}, "
        f"$L$={CFG['L']}, $\\theta_E$={CFG['theta_E_deg']}°, "
        f"$n_{{trials}}$={CFG['n_trials']})",
        fontsize=10,
    )
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xlim([snr_db[0], snr_db[-1]])
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Saved -> {save_path}")

    return fig


def plot_crb_vs_L(
    L_vals:   NDArray,
    crb_deg:  NDArray,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot theoretical sqrt(CRB) vs number of sensing beams L."""
    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(7, 4.8))

    ax.plot(L_vals, crb_deg,     "b-o",  ms=6,  label=r"$\sqrt{\mathrm{CRB}}$")
    ax.plot(L_vals, 3 * crb_deg, "r--s", ms=5,
            label=r"$3\sqrt{\mathrm{CRB}}$ (99.7% confidence)")

    # Mark our default L=32
    idx32 = np.where(L_vals == 32)[0]
    if len(idx32):
        ax.axvline(32, color="gray", linestyle=":", lw=1.2, label="Default $L=32$")

    ax.set_xlabel("Number of sensing beams $L$", fontsize=11)
    ax.set_ylabel("Angle estimation error [degrees]", fontsize=11)
    ax.set_title(
        f"CRB vs Number of Sensing Beams\n"
        f"($M_t$={CFG['M_t']}, $M_r$={CFG['M_r']}, "
        f"$\\theta_E$={CFG['theta_E_deg']}°, "
        f"$P_s$={CFG['P_s_dBm']} dBm, $\\sigma^2$={CFG['sigma2_dBm']} dBm)",
        fontsize=10,
    )
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xlim([L_vals[0], L_vals[-1]])
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    print("=" * 55)
    print("  Sensing Stage Validation")
    print("=" * 55)
    print(f"  M_t={CFG['M_t']}, M_r={CFG['M_r']}, L={CFG['L']}")
    print(f"  theta_E={CFG['theta_E_deg']} deg")
    print(f"  P_s={CFG['P_s_dBm']} dBm, sigma2={CFG['sigma2_dBm']} dBm")
    print(f"  d_be={CFG['d_be']} m, f_c={CFG['f_c']/1e9:.0f} GHz")
    print()

    # Test 1
    snr_db, rmse_deg, crb_deg = test_rmse_vs_snr()
    fig1 = plot_rmse_vs_snr(
        snr_db, rmse_deg, crb_deg,
        save_path=os.path.join(OUT, "test1_rmse_vs_snr.png"),
    )
    plt.close(fig1)

    # Test 2
    L_vals, crb_L = test_crb_vs_L()
    fig2 = plot_crb_vs_L(
        L_vals, crb_L,
        save_path=os.path.join(OUT, "test2_crb_vs_L.png"),
    )
    plt.close(fig2)

    print("\n  All tests complete.")
    print(f"  Figures saved to: {OUT}")
