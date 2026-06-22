# reproduce_fig2.py
# Reproduces Fig. 2 from Su et al. (TWC 2024)
# Spatial spectral estimates with CAML approach

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from sensing import (steering_vector, compute_RX, simulate_echo_multi,
                     capon_spectrum, aml_amplitudes)

# === Paper parameters (Section IV, Fig. 2) ===
N_t = 10
N_r = 10
L   = 64
P0  = 1.0

# Eve angles and amplitudes (unknown to BS)
theta_eve  = [-25.0, 15.0]
beta_mags  = [1.0, 5.0]

# CU angles and amplitudes (KNOWN to BS — not in echo)
theta_cu   = [40.0, 10.0, -30.0]
beta_cu    = [4.0,  4.0,   2.0]

# Fixed random phases for beta
rng_beta  = np.random.default_rng(42)
beta_list = [m * np.exp(1j * rng_beta.uniform(0, 2*np.pi))
             for m in beta_mags]

# SNR definition: SNR = |beta_min|^2 * N_r * (P0/N_t) / sigma2_R
# beta_min = 1.0 (weakest Eve)
def sigma2_from_snr(snr_dB, N_t=10, N_r=10, P0=1.0, beta_ref=1.0):
    snr_lin = 10**(snr_dB / 10.0)
    return (beta_ref**2 * N_r * P0 / N_t) / snr_lin

theta_grid = np.linspace(-60.0, 60.0, 2401)  # 0.05 deg resolution

# === Plot ===
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle(
    "Fig. 2 — Spatial spectral estimates with CAML approach\n"
    r"Eves: $\theta_1=-25°,\ \theta_2=15°$ (blue)  |  "
    r"CUs: $\theta_3=40°,\ \theta_4=10°,\ \theta_5=-30°$ (green)",
    fontsize=10, y=1.02
)

for ax, snr_dB, panel in zip(axes, [20.0, -15.0], ["(a)", "(b)"]):

    sigma2_R = sigma2_from_snr(snr_dB)

    # simulate echo — Eve reflections only
    Y, X = simulate_echo_multi(
        theta_eve_deg = theta_eve,
        beta_list     = beta_list,
        N_t = N_t, N_r = N_r, L = L,
        P0  = P0, sigma2_R = sigma2_R,
        seed = 0,
    )

    # Step 1: Capon → find K=2 Eve angle estimates
    spec   = capon_spectrum(Y, N_r, theta_grid)
    peaks, _ = find_peaks(spec, height=0.01*spec.max(), distance=20)
    top2   = peaks[np.argsort(spec[peaks])[::-1][:2]]
    theta_hat = sorted(theta_grid[top2])

    # Step 2: AML → amplitude estimates at estimated angles
    beta_hat = aml_amplitudes(Y, X, theta_hat, N_r, N_t)
    beta_hat_mag = np.abs(beta_hat)

    # --- plot estimated Eves — blue solid stems ---
    for th, bm in zip(theta_hat, beta_hat_mag):
        ax.vlines(th, 0, bm, colors="royalblue", linewidth=2.5)
        ax.plot(th, bm, "o", color="royalblue", markersize=5)

    # --- plot known CUs — green solid stems ---
    for th, bm in zip(theta_cu, beta_cu):
        ax.vlines(th, 0, bm, colors="forestgreen", linewidth=2.5)
        ax.plot(th, bm, "o", color="forestgreen", markersize=5)

    # --- true Eve directions — red dashed (panel b only) ---
    if snr_dB < 0:
        for th, bm in zip(theta_eve, beta_mags):
            ax.vlines(th, 0, bm, colors="red",
                      linewidth=1.8, linestyle="--")
            ax.plot(th, bm, "x", color="red", markersize=6)

    # --- labels and formatting ---
    ax.set_title(f"{panel} SNR={snr_dB:.0f} dB", fontsize=10)
    ax.set_xlabel("DOA (deg)", fontsize=9)
    ax.set_ylabel("Modulus of Complex Amplitude", fontsize=9)
    ax.set_xlim(-60, 60)
    ax.set_ylim(0, 6)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    # legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], color="royalblue",    linewidth=2, label="Eve (estimated)"),
        Line2D([0],[0], color="forestgreen",  linewidth=2, label="CU (known)"),
    ]
    if snr_dB < 0:
        legend_elements.append(
            Line2D([0],[0], color="red", linewidth=1.8,
                   linestyle="--", label="Eve (true)")
        )
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

    # print estimates
    print(f"\n{'='*40}")
    print(f"SNR = {snr_dB:.0f} dB")
    for k in range(2):
        print(f"  Eve {k+1}: true=({theta_eve[k]:.1f}°, |β|={beta_mags[k]:.1f})  "
              f"est=({theta_hat[k]:.1f}°, |β̂|={beta_hat_mag[k]:.3f})")

plt.tight_layout()

out_path = Path(__file__).parent / "outputs" / "fig2_caml_stem.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out_path}")
plt.close()