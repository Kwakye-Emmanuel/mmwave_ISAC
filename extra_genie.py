"""Run extra Genie-Aided trials at specific SNR points
and update saved NPZ results for clean outage curve."""
import numpy as np
from joblib import Parallel, delayed
from isac.config import SystemConfig
from isac.channel import generate_channels
from isac.scheduling import oracle_scheduling_genie
from isac.simulate import plot_outage_vs_snr, _topology_tag
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────
cfg       = SystemConfig()
tag       = _topology_tag(cfg)
npz_path  = f"outputs/eval_results{tag}.npz"
snr_extra = [5.0, 7.5]      # SNR points to fix
n_trials  = 10_000_000      # 10M trials

# ── Trial worker ──────────────────────────────────────
def run_trial(t, snr):
    seed = cfg.seed + t + int(snr * 10) * 100_000
    P_t  = 10 ** (snr / 10.0) * cfg.sigma2_C

    sample = generate_channels(
        M=cfg.M, N=cfg.N,
        d_0=cfg.d_0, d_be=cfg.d_be,
        eta=cfg.eta,
        d_cu_min=cfg.d_cu_min,
        d_cu_max=cfg.d_cu_max,
        beta_e_mag=cfg.beta_e_mag,
        theta_E_min=cfg.theta_E_min_rad,
        theta_E_max=cfg.theta_E_max_rad,
        seed=seed,
    )
    H, g_e = sample["H"], sample["g_e"]

    _, r_ora = oracle_scheduling_genie(
        H, g_e, cfg.Kd, P_t,
        cfg.sigma2_C, cfg.time_frac, cfg.rho)

    return float(r_ora < cfg.R0)

# ── Load existing results ─────────────────────────────
print(f"\n  Loading {npz_path} ...")
data        = np.load(npz_path)
snr_db      = data['snr_db']
pout_oracle = data['pout_oracle'].copy()

print(f"  SNR points : {snr_db}")
print(f"  pout_oracle (before) : {pout_oracle}")

# ── Run extra trials ──────────────────────────────────
for snr in snr_extra:
    # Find index of this SNR point
    idx = np.argmin(np.abs(snr_db - snr))
    print(f"\n  Running {n_trials:,} trials at "
          f"SNR={snr}dB (index={idx}) ...")

    results = Parallel(n_jobs=-1)(
        delayed(run_trial)(t, snr)
        for t in range(n_trials)
    )
    pout_new = float(np.mean(results))
    print(f"  SNR={snr}dB | "
          f"Genie pout = {pout_new:.3e}")

    # Only update if new value is more informative
    if pout_new > 0:
        pout_oracle[idx] = pout_new
        print(f"  Updated pout_oracle[{idx}] = "
              f"{pout_new:.3e}")
    else:
        print(f"  Still zero at {n_trials:,} trials "
              f"— keeping original")

print(f"\n  pout_oracle (after) : {pout_oracle}")

# ── Resave NPZ ────────────────────────────────────────
np.savez(
    npz_path,
    snr_db        = snr_db,
    rsec_B1       = data['rsec_B1'],
    rsec_B2       = data['rsec_B2'],
    rsec_oracle   = data['rsec_oracle'],
    rsec_proposed = data['rsec_proposed'],
    rsec_conv_dl  = data['rsec_conv_dl'],
    rsec_deepsets = data['rsec_deepsets'],
    pout_B1       = data['pout_B1'],
    pout_B2       = data['pout_B2'],
    pout_oracle   = pout_oracle,      # ← updated
    pout_proposed = data['pout_proposed'],
    pout_conv_dl  = data['pout_conv_dl'],
)
print(f"\n  Resaved -> {npz_path}")

# ── Replot ────────────────────────────────────────────
def nan_zeros(arr):
    out = arr.copy().astype(float)
    out[out == 0.0] = np.nan
    return out

for ext in ["png", "pdf"]:
    fig = plot_outage_vs_snr(
        snr_db,
        nan_zeros(data['pout_B1']),
        nan_zeros(data['pout_B2']),
        nan_zeros(pout_oracle),
        pout_proposed = nan_zeros(data['pout_proposed']),
        pout_conv_dl  = nan_zeros(data['pout_conv_dl']),
        cfg           = cfg,
        save_path     = f"outputs/replot_outage_vs_snr{tag}.{ext}",
    )
    plt.close(fig)

print(f"  Replotted -> replot_outage_vs_snr{tag}.pdf")
print("\n  Done.")
