"""Generate plots for meeting.

Plots saved to outputs/:
    1. loss_curve.png          — GFlowNet TB loss over episodes
    2. secrecy_rate_vs_snr.png — secrecy sum-rate vs SNR (random W vs ZF + AN)
"""
from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# ----------------------------------------------------------------
# Channel / beamforming / signal stubs
# ----------------------------------------------------------------
def steering_vector(angle_deg, N):
    indices = np.arange(N) - (N - 1) / 2
    return np.exp(-1j * np.pi * indices * np.sin(np.deg2rad(angle_deg)))

def generate_cu_channel(N_t, kappa, L_p, angle_los_deg, rng):
    h_los  = np.sqrt(N_t) * steering_vector(angle_los_deg, N_t)
    h_nlos = np.zeros(N_t, dtype=complex)
    for _ in range(L_p):
        c  = (rng.standard_normal() + 1j*rng.standard_normal()) / np.sqrt(2)
        h_nlos += c * steering_vector(np.degrees(rng.uniform(-np.pi/2, np.pi/2)), N_t)
    h_nlos = np.sqrt(N_t / L_p) * h_nlos
    return np.sqrt(kappa/(1+kappa))*h_los + np.sqrt(1/(1+kappa))*h_nlos

def generate_channels_mmwave(N_t, N, kappa, L_p, sigma_alpha,
                              theta_E_min_deg, theta_E_max_deg, seed=None):
    rng     = np.random.default_rng(seed)
    theta_E = float(rng.uniform(theta_E_min_deg, theta_E_max_deg))
    H = np.zeros((N_t, N), dtype=complex)
    for i in range(N):
        H[:, i] = generate_cu_channel(N_t, kappa, L_p,
                                       float(rng.uniform(-90, 90)), rng)
    alpha_k = (rng.standard_normal() + 1j*rng.standard_normal()) / np.sqrt(2) * sigma_alpha
    g_e     = alpha_k * steering_vector(theta_E, N_t)
    return {"H": H, "g_e": g_e, "theta_E": theta_E}

def zf_precoder(H_K, P_t, phi, K):
    gram = H_K.conj().T @ H_K
    W    = H_K @ np.linalg.inv(gram)
    p    = np.sqrt(phi * P_t / K)
    for k in range(K):
        n = np.linalg.norm(W[:, k])
        if n > 1e-12:
            W[:, k] = W[:, k] / n * p
    return W

def null_space_projector(H_K, N_t):
    gram = H_K.conj().T @ H_K
    return np.eye(N_t) - H_K @ np.linalg.inv(gram) @ H_K.conj().T

def an_covariance_isotropic(H_K, N_t, P_t, phi, K):
    V = null_space_projector(H_K, N_t)
    return ((1 - phi) * P_t / (N_t - K)) * V

def compute_sinr_cu(h_i, W, R_N, i, sigma2_C):
    I      = W.shape[1]
    sig    = abs(h_i.conj() @ W[:, i]) ** 2
    interf = sum(abs(h_i.conj() @ W[:, m])**2 for m in range(I) if m != i)
    an     = float(np.real(h_i.conj() @ R_N @ h_i))
    return sig / (interf + an + sigma2_C)

def compute_sinr_eve(g_e, W, R_N, i, sigma2_e):
    I   = W.shape[1]
    sig = float(np.real(g_e.conj() @ (W[:, i, None] @ W[:, i, None].conj().T) @ g_e))
    Im  = sum(W[:, m, None] @ W[:, m, None].conj().T for m in range(I) if m != i) + R_N
    den = float(np.real(g_e.conj() @ Im @ g_e)) + sigma2_e
    return sig / den

def compute_secrecy_rate(sinr_cu, sinr_eve):
    return float(sum(max(0.0, np.log2(1+sc) - np.log2(1+se))
                     for sc, se in zip(sinr_cu, sinr_eve)))

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
N_t         = 10
N_users     = 10
K           = 2
kappa       = 0.1
L_p         = 3
sigma_alpha = 1.0
theta_E_min = -30.0
theta_E_max =  30.0
sigma2_C    = 1.0
sigma2_e    = 1.0
rho         = 0.5
n_trials    = 1000
snr_dB_range = np.linspace(0, 20, 9)
seed        = 42

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

rng_global = np.random.default_rng(seed)

# ----------------------------------------------------------------
# IEEE paper style
# ----------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    "axes.grid":            True,
    "grid.color":           "#cccccc",
    "grid.linewidth":       0.6,
    "grid.linestyle":       "--",
    "axes.spines.top":      True,
    "axes.spines.right":    True,
    "axes.spines.left":     True,
    "axes.spines.bottom":   True,
    "axes.linewidth":       0.8,
    "font.family":          "serif",
    "font.size":            12,
    "axes.labelsize":       13,
    "axes.titlesize":       13,
    "legend.fontsize":      11,
    "xtick.labelsize":      11,
    "ytick.labelsize":      11,
    "lines.linewidth":      1.8,
    "lines.markersize":     7,
})

# ----------------------------------------------------------------
# Eval functions
# ----------------------------------------------------------------
def eval_random(ch, P_t, rng):
    H, g_e = ch["H"], ch["g_e"]
    best = 0.0
    for sel in combinations(range(N_users), K):
        W   = (rng.standard_normal((N_t, K)) +
               1j*rng.standard_normal((N_t, K))) / np.sqrt(2)
        W   = W / np.linalg.norm(W, 'fro') * np.sqrt(rho * P_t)
        R_N = ((1 - rho) * P_t / N_t) * np.eye(N_t)
        sc  = [compute_sinr_cu(H[:, sel[i]], W, R_N, i, sigma2_C) for i in range(K)]
        se  = [compute_sinr_eve(g_e, W, R_N, i, sigma2_e) for i in range(K)]
        r   = compute_secrecy_rate(sc, se)
        if r > best: best = r
    return best

def eval_zf(ch, P_t):
    H, g_e = ch["H"], ch["g_e"]
    best = 0.0
    for sel in combinations(range(N_users), K):
        H_K = H[:, list(sel)]
        W   = zf_precoder(H_K, P_t, rho, K)
        R_N = an_covariance_isotropic(H_K, N_t, P_t, rho, K)
        sc  = [compute_sinr_cu(H[:, sel[i]], W, R_N, i, sigma2_C) for i in range(K)]
        se  = [compute_sinr_eve(g_e, W, R_N, i, sigma2_e) for i in range(K)]
        r   = compute_secrecy_rate(sc, se)
        if r > best: best = r
    return best

# ----------------------------------------------------------------
# Plot 1 — GFlowNet loss curve
# ----------------------------------------------------------------
print("Training GFlowNet for loss curve (10k episodes)...")

class _PolicyNet(nn.Module):
    def __init__(self, N, hidden):
        super().__init__()
        self.N   = N
        self.net = nn.Sequential(
            nn.Linear(2*N, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2*N),
        )
    def forward(self, s, snr):
        out = self.net(torch.cat([s, snr], dim=-1))
        return out[..., :self.N], out[..., self.N:]

def masked_log_softmax(logits, mask):
    logits = logits.clone(); logits[~mask] = -1e9
    return F.log_softmax(logits, dim=-1)

policy    = _PolicyNet(N_users, 256)
log_z     = nn.Parameter(torch.zeros(1))
opt       = Adam(list(policy.parameters()) + [log_z], lr=1e-3)
losses    = []
n_ep      = 10000
temp_start, temp_end = 2.0, 0.1
rng_train = np.random.default_rng(seed)

for ep in range(n_ep):
    ch    = generate_channels_mmwave(N_t, N_users, kappa, L_p, sigma_alpha,
                                     theta_E_min, theta_E_max,
                                     seed=int(rng_train.integers(0, 2**31)))
    norms = np.array([np.sum(np.abs(ch["H"][:, i])**2) for i in range(N_users)])
    snr   = torch.tensor(norms, dtype=torch.float32)
    frac  = min(ep / (n_ep * 0.8), 1.0)
    temp  = max(temp_start + frac*(temp_end - temp_start), 1e-6)

    s = torch.zeros(N_users); transitions = []
    for _ in range(K):
        fwd, _ = policy(s.unsqueeze(0), snr.unsqueeze(0))
        fwd    = fwd.squeeze(0) / temp
        lpf    = masked_log_softmax(fwd, s < 0.5)
        a      = int(torch.multinomial(lpf.exp(), 1).item())
        s_next = s.clone(); s_next[a] = 1.0
        transitions.append((s.clone(), a, s_next.clone()))
        s = s_next

    selected = tuple(sorted(t[1] for t in transitions))
    reward   = eval_zf(ch, P_t=1.0)

    K_      = len(transitions)
    snr_exp = snr.unsqueeze(0).expand(K_, -1)
    s_b     = torch.stack([t[0] for t in transitions])
    sn_b    = torch.stack([t[2] for t in transitions])
    fwd_b, _  = policy(s_b,  snr_exp)
    _,  bwd_b = policy(sn_b, snr_exp)
    fwd_b = fwd_b / temp
    lpf = torch.zeros(1); lpb = torch.zeros(1)
    for i, (s_, a_, sn_) in enumerate(transitions):
        lpf = lpf + masked_log_softmax(fwd_b[i], s_  < 0.5)[a_]
        lpb = lpb + masked_log_softmax(bwd_b[i], sn_ > 0.5)[a_]
    log_r = math.log(max(reward, 1e-8))
    loss  = (log_z + lpf - log_r - lpb).pow(2)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())

    if (ep+1) % 1000 == 0:
        print(f"  ep {ep+1:>6d}  loss={np.mean(losses[-1000:]):.4f}")

window   = 200
smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
eps_s    = np.arange(window, len(losses)+1)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(eps_s, smoothed, color="#0055A4", linewidth=1.8,
            label="TB loss (200-ep moving avg)")
ax.axvline(x=4000, color="#CC3311", linewidth=1.2, linestyle="--",
           label="Convergence point (~ep 4k)")
ax.set_xlabel("Episode")
ax.set_ylabel("Trajectory Balance Loss")
ax.set_title("GFlowNet Training Convergence")
ax.legend(frameon=True, edgecolor="#cccccc", fancybox=False)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
fig.tight_layout()
fig.savefig(OUT / "loss_curve.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT / 'loss_curve.png'}")

# ----------------------------------------------------------------
# Plot 2 — Secrecy rate vs SNR
# ----------------------------------------------------------------
print(f"\nComputing secrecy rate vs SNR ({len(snr_dB_range)} pts, {n_trials} trials)...")

sr_random = []
sr_zf     = []

for snr_dB in snr_dB_range:
    P_t  = 10 ** (snr_dB / 10.0)
    r_rnd, r_zf = [], []
    for _ in range(n_trials):
        ch = generate_channels_mmwave(
            N_t, N_users, kappa, L_p, sigma_alpha,
            theta_E_min, theta_E_max,
            seed=int(rng_global.integers(0, 2**31)),
        )
        r_rnd.append(eval_random(ch, P_t, rng_global))
        r_zf.append(eval_zf(ch, P_t))
    sr_random.append(np.mean(r_rnd))
    sr_zf.append(np.mean(r_zf))
    print(f"  SNR={snr_dB:4.0f} dB  Random={sr_random[-1]:.3f}  ZF+AN={sr_zf[-1]:.3f}")

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(snr_dB_range, sr_random,
        "o--", color="#CC3311", linewidth=1.8, markersize=7,
        markerfacecolor="white", markeredgewidth=1.8,
        label="RS with isotropic AN (random beamforming)")

ax.plot(snr_dB_range, sr_zf,
        "s-", color="#0055A4", linewidth=1.8, markersize=7,
        markerfacecolor="white", markeredgewidth=1.8,
        label="Proposed (ZF + null-space AN)")

ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Secrecy Sum-Rate (bits/s/Hz)")
ax.set_title(f"Secrecy Sum-Rate vs. SNR\n"
             f"($N_t={N_t}$, $N={N_users}$ users, $K={K}$ scheduled, $\\kappa={kappa}$)")
ax.legend(frameon=True, edgecolor="#cccccc", fancybox=False, loc="upper left")
ax.set_xlim([snr_dB_range[0], snr_dB_range[-1]])
ax.set_ylim(bottom=0)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))

fig.tight_layout()
fig.savefig(OUT / "secrecy_rate_vs_snr.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT / 'secrecy_rate_vs_snr.png'}")

print("\nDone. Check outputs/ folder.")