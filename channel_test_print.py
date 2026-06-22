"""Quick diagnostic to compare g_hat_e constructions."""
import numpy as np
from isac.config import SystemConfig
from isac.channel import generate_channels
from isac.sensing import compute_beta_s, run_sensing, ula_steering

cfg        = SystemConfig()
trial_seed = 42
rng        = np.random.default_rng(trial_seed)

# ---------------------------------------------------------------------------
# 1. Generate channels
# ---------------------------------------------------------------------------
sample  = generate_channels(
    M=cfg.M, N=cfg.N, d_0=cfg.d_0, d_be=cfg.d_be,
    eta=cfg.eta, d_cu_min=cfg.d_cu_min, d_cu_max=cfg.d_cu_max,
    sigma_e=cfg.sigma_e,
    theta_E_min=cfg.theta_E_min_rad,
    theta_E_max=cfg.theta_E_max_rad,
    seed=trial_seed,
)
H, g_e, theta_E = sample["H"], sample["g_e"], sample["theta_E"]

# ---------------------------------------------------------------------------
# 2. Run sensing
# ---------------------------------------------------------------------------
beta_s_mag = compute_beta_s(cfg.d_be, cfg.f_c, cfg.epsilon_dBsm)
beta_s_t   = beta_s_mag * np.exp(1j * rng.uniform(0, 2 * np.pi))
state      = run_sensing(
    theta_E, cfg.M, cfg.M_r, cfg.L,
    cfg.P_s, cfg.sigma2_s, beta_s_t, seed=trial_seed,
)

# ---------------------------------------------------------------------------
# 3. Three versions of g_hat_e
# ---------------------------------------------------------------------------
g_hat_e_old  = state["beta_hat"] * state["at_hat"]   # old: radar LS
g_hat_e_new  = cfg.sigma_e * state["at_hat"]          # new: expected value
g_hat_e_genie = g_e                                    # genie: perfect CSI

# ---------------------------------------------------------------------------
# 4. Print comparison
# ---------------------------------------------------------------------------
print("=" * 60)
print("  ANGLE ESTIMATION")
print("=" * 60)
print(f"  True theta_E  : {np.degrees(theta_E):.4f} deg")
print(f"  theta_hat     : {np.degrees(state['theta_hat']):.4f} deg")
print(f"  Error         : {np.degrees(abs(theta_E - state['theta_hat'])):.4f} deg")
print(f"  CRB (std dev) : {np.degrees(np.sqrt(state['crb'])):.4f} deg")

print()
print("=" * 60)
print("  CHANNEL POWER COMPARISON  (should be ~M=8 on average)")
print("=" * 60)
print(f"  True g_e power       : {np.sum(np.abs(g_e)**2):.4f}")
print(f"  g_hat_e OLD power    : {np.sum(np.abs(g_hat_e_old)**2):.4f}")
print(f"  g_hat_e NEW power    : {np.sum(np.abs(g_hat_e_new)**2):.4f}")
print(f"  Genie power          : {np.sum(np.abs(g_hat_e_genie)**2):.4f}")

print()
print("=" * 60)
print("  BETA VALUES")
print("=" * 60)
print(f"  True |alpha_e|       : {abs(g_e[0] / ula_steering(theta_E, cfg.M)[0]):.4f}")
print(f"  |beta_hat| (radar)   : {abs(state['beta_hat']):.6f}")
print(f"  cfg.sigma_e          : {cfg.sigma_e:.4f}")

print()
print("=" * 60)
print("  DIRECTION ALIGNMENT")
print("=" * 60)
a_true = ula_steering(theta_E, cfg.M)
a_hat  = state["at_hat"]
print(f"  |a(theta_E)^H a(theta_hat)| / M : "
      f"{abs(a_true.conj() @ a_hat) / cfg.M:.4f}")
print("  (1.0 = perfect alignment, <1.0 = estimation error)")