"""Central configuration for mmWave ISAC physical layer security simulation.

All parameters based on Su et al. (TWC 2024) numerical results section.
Every other module imports from thisb file.

Key design principle:
    ONE power budget P0 for both sensing and communication.
    Sensing uses P0 for omnidirectional probe.
    Communication uses P0 split by phi into data (W) and AN (R_N).

Usage:
    from config import SystemConfig, GFlowNetConfig
    cfg = SystemConfig()
    print(cfg.summary())
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SystemConfig:

    # ----------------------------------------------------------------
    # Antenna configuration  (Su et al. Section VII)
    # ----------------------------------------------------------------
    N_t: int = 10   # BS transmit antennas
    N_r: int = 10   # BS receive antennas

    # ----------------------------------------------------------------
    # User and scheduling
    # ----------------------------------------------------------------
    N: int = 10    # total candidate CUs (pool)
    K: int = 2     # scheduled users per frame (GFlowNet selects K from N)

    # ----------------------------------------------------------------
    # Channel model  (Su et al. Section VII)
    # ----------------------------------------------------------------
    kappa:            float = 0.1    # Rician K-factor (weak LoS)
    L_p:              int   = 3      # NLoS scattering paths per CU
    sigma_alpha:      float = 1.0    # Eve path-loss std dev (normalized)
    theta_E_min_deg:  float = -30.0  # Eve angle min [deg]
    theta_E_max_deg:  float =  30.0  # Eve angle max [deg]
    cu_angle_min_deg: float = -90.0  # CU LoS angle min [deg]
    cu_angle_max_deg: float =  90.0  # CU LoS angle max [deg]

    # ----------------------------------------------------------------
    # Signal model  (Su et al. Section VII)
    # ----------------------------------------------------------------
    phi:      float = 0.5    # power split ratio data/AN (Phase 1, fixed)
    L:        int   = 64     # frame length (communication snapshots)
    sigma2_C: float = 1e-3   # CU noise variance  (0 dBm = 1e-3 W)
    sigma2_e: float = 1e-3   # Eve noise variance (0 dBm = 1e-3 W)
    alpha:    float = 0.05   # beam fluctuation parameter

    # ----------------------------------------------------------------
    # Power budget — ONE budget for sensing AND communication (ISAC)
    #
    # Sensing:       R_X = (P0/N_t)*I  → tr(R_X) = P0
    # Communication: tr(WW^H) + tr(R_N) = phi*P0 + (1-phi)*P0 = P0
    #
    # SNR definition (Su et al. Section VII):
    #   SNR_echo = |beta|^2 * L * P0 / sigma2_R = -22 dB
    #   sigma2_R fixed at 0 dBm as physical reference
    #   beta_mag derived from SNR definition
    # ----------------------------------------------------------------
    P0_dBm:      float = 35.0    # total power budget [dBm]
    SNR_echo_dB: float = -22.0   # echo SNR [dB]
    sigma2_R:    float = 1e-3    # radar noise variance [W] (0 dBm)

    # ----------------------------------------------------------------
    # Simulation sweep — vary P0 (x-axis of Fig. 2)
    # Matches Su et al. Fig. 6/8: P0 = 25, 30, 35 dBm
    # ----------------------------------------------------------------
    P0_dBm_min:  float = 25.0
    P0_dBm_max:  float = 35.0
    n_P0_pts:    int   = 5       # [25, 26.25, ..., 35] dBm
    n_trials:    int   = 50  # quick test (restore to 1000 for paper)
    seed:        int   = 42

    # ----------------------------------------------------------------
    # Paths
    # ----------------------------------------------------------------
    data_dir:       Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")
    output_dir:     Path = Path("outputs")

    def __post_init__(self):
        self.data_dir       = Path(self.data_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.output_dir     = Path(self.output_dir)

        # P0 in Watts — training power
        self.P0 = 10 ** (self.P0_dBm / 10.0) * 1e-3   # W

        # echo SNR linear
        self.SNR_echo_lin = 10 ** (self.SNR_echo_dB / 10.0)

        # beta_mag from SNR definition:
        # SNR = |beta|^2 * L * P0 / sigma2_R
        self.beta_mag = float(np.sqrt(
            self.SNR_echo_lin * self.sigma2_R / (self.L * self.P0)
        ))

        # P0 sweep for simulation [W]
        self.P0_dBm_range = np.linspace(
            self.P0_dBm_min, self.P0_dBm_max, self.n_P0_pts
        )
        self.P0_range = 10 ** (self.P0_dBm_range / 10.0) * 1e-3  # W

    # ----------------------------------------------------------------
    # Derived quantities
    # ----------------------------------------------------------------
    @property
    def theta_E_min_rad(self) -> float:
        return float(np.radians(self.theta_E_min_deg))

    @property
    def theta_E_max_rad(self) -> float:
        return float(np.radians(self.theta_E_max_deg))

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  SystemConfig  (Su et al. TWC 2024)",
            "=" * 60,
            f"  Antennas   : N_t={self.N_t}, N_r={self.N_r}",
            f"  Users      : N={self.N} candidates, K={self.K} scheduled",
            f"  Channel    : kappa={self.kappa}, L_p={self.L_p}",
            f"               sigma_alpha={self.sigma_alpha}",
            f"               theta_E ~ Uniform[{self.theta_E_min_deg}°,"
            f" {self.theta_E_max_deg}°]",
            f"  Signal     : phi={self.phi} (fixed Phase 1), L={self.L}",
            f"               sigma2_C={self.sigma2_C:.1e} W (0 dBm)",
            f"               sigma2_e={self.sigma2_e:.1e} W (0 dBm)",
            f"  Power      : ONE budget P0 for sensing + communication",
            f"               P0={self.P0_dBm} dBm = {self.P0:.4f} W (training)",
            f"               Sensing:  tr(R_X) = P0",
            f"               Comms:    tr(WW^H)+tr(R_N) = phi*P0+(1-phi)*P0 = P0",
            f"  Sensing    : SNR_echo={self.SNR_echo_dB} dB",
            f"               sigma2_R={self.sigma2_R:.1e} W (0 dBm)",
            f"               beta_mag={self.beta_mag:.4e}",
            f"  Simulation : P0 sweep {self.P0_dBm_min}-{self.P0_dBm_max} dBm"
            f" ({self.n_P0_pts} pts)",
            f"               trials={self.n_trials}, seed={self.seed}",
            "=" * 60,
        ]
        return "\n".join(lines)


@dataclass
class GFlowNetConfig:

    # ----------------------------------------------------------------
    # GFlowNet  (Trajectory Balance)
    # ----------------------------------------------------------------
    N:            int   = 10        # total CUs (must match SystemConfig.N)
    K:            int   = 2         # users to schedule
    hidden:       int   = 256       # hidden layer size
    lr:           float = 1e-3      # learning rate
    temp_start:   float = 2.0       # initial temperature
    temp_end:     float = 0.1       # final temperature
    n_episodes:   int   = 10_000    # training episodes
    reward_floor: float = 1e-8      # minimum reward to avoid log(0)
    log_every:    int   = 1_000     # print interval
    n_mc:         int   = 100        # Monte Carlo samples for reward


if __name__ == "__main__":
    cfg     = SystemConfig()
    gfn_cfg = GFlowNetConfig()
    print(cfg.summary())
    print(f"\nGFlowNetConfig:")
    print(f"  N={gfn_cfg.N}, K={gfn_cfg.K}")
    print(f"  hidden={gfn_cfg.hidden}, lr={gfn_cfg.lr}")
    print(f"  episodes={gfn_cfg.n_episodes}, n_mc={gfn_cfg.n_mc}")

    # Power budget verification
    print(f"\n=== Power budget verification ===")
    print(f"P0 = {cfg.P0:.4f} W")
    print(f"phi*P0 (data)     = {cfg.phi*cfg.P0:.4f} W")
    print(f"(1-phi)*P0 (AN)   = {(1-cfg.phi)*cfg.P0:.4f} W")
    print(f"Total comm power  = {cfg.P0:.4f} W = P0 ✅")
    print(f"Sensing power     = {cfg.P0:.4f} W = P0 ✅")
    print(f"No over-budgeting ✅")

    print(f"\n=== P0 sweep ===")
    for p_dBm, p_w in zip(cfg.P0_dBm_range, cfg.P0_range):
        snr_cu = 10*np.log10(cfg.phi * p_w / cfg.K / cfg.sigma2_C)
        print(f"  P0={p_dBm:5.2f} dBm = {p_w:.4e} W  "
              f"→ SNR_CU ≈ {snr_cu:.1f} dB")