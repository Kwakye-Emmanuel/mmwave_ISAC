"""Central configuration for mmWave ISAC physical layer security simulation.

All parameters based on CAML paper (Su et al.) numerical results section.
Every other module imports from this file — no scattered magic numbers.

Usage:
    from config import SystemConfig
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
    # Antenna configuration  (paper Section VII)
    # ----------------------------------------------------------------
    N_t: int = 10   # BS transmit antennas
    N_r: int = 10   # BS receive antennas

    # ----------------------------------------------------------------
    # User and scheduling  (paper Section VII)
    # ----------------------------------------------------------------
    N: int = 10    # total CUs
    K: int = 2     # scheduled users per frame

    # ----------------------------------------------------------------
    # Channel model  (CAML paper Eq. 2, Section VII)
    # ----------------------------------------------------------------
    kappa:            float = 0.1    # Rician K-factor (weak LoS)
    L_p:              int   = 3      # NLoS scattering paths per CU
    sigma_alpha:      float = 1.0    # Eve path-loss std dev
    theta_E_min_deg:  float = -30.0  # Eve angle min [deg]
    theta_E_max_deg:  float =  30.0  # Eve angle max [deg]
    cu_angle_min_deg: float = -90.0  # CU LoS angle min [deg]
    cu_angle_max_deg: float =  90.0  # CU LoS angle max [deg]

    # ----------------------------------------------------------------
    # Signal model  (paper Section VII)
    # ----------------------------------------------------------------
    P_t:      float = 1.0   # total transmit power [W]
    sigma2_C: float = 1.0   # CU noise variance (0 dBm normalised)
    sigma2_e: float = 1.0   # Eve noise variance
    rho:      float = 0.5   # power split ratio (data / AN)
    L:        int   = 64    # frame length (paper Section VII)

    # ----------------------------------------------------------------
    # Simulation
    # ----------------------------------------------------------------
    n_trials: int = 10000
    seed:     int = 42


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

    # ----------------------------------------------------------------
    # Derived quantities
    # ----------------------------------------------------------------

    @property
    def theta_E_min_rad(self) -> float:
        """Eve min angle [rad]."""
        return float(np.radians(self.theta_E_min_deg))

    @property
    def theta_E_max_rad(self) -> float:
        """Eve max angle [rad]."""
        return float(np.radians(self.theta_E_max_deg))

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  SystemConfig  (CAML paper — Su et al.)",
            "=" * 60,
            f"  Antennas   : N_t={self.N_t}, N_r={self.N_r}",
            f"  Users      : N={self.N}, K={self.K}",
            f"  Channel    : kappa={self.kappa}, L_p={self.L_p}",
            f"               sigma_alpha={self.sigma_alpha}",
            f"               theta_E ~ Uniform[{self.theta_E_min_deg}°, {self.theta_E_max_deg}°]",
            f"  Signal     : P_t={self.P_t}, rho={self.rho}, L={self.L}",
            f"               sigma2_C={self.sigma2_C}, sigma2_e={self.sigma2_e}",
            f"  Simulation : trials={self.n_trials}, seed={self.seed}",
            "=" * 60,
        ]
        return "\n".join(lines)


@dataclass
class GFlowNetConfig:
    # ----------------------------------------------------------------
    # GFlowNet  (to be updated after full discussion)
    # ----------------------------------------------------------------
    N:            int   = 10       # total CUs (must match SystemConfig.N)
    K:            int   = 2        # users to schedule (must match SystemConfig.K)
    hidden:       int   = 256      # hidden layer size
    lr:           float = 1e-3     # learning rate
    temp_start:   float = 2.0      # initial temperature
    temp_end:     float = 0.1      # final temperature
    n_episodes:   int   = 50_000   # episodes for initial testing
    reward_floor: float = 1e-8     # minimum reward to avoid log(0)
    log_every:    int   = 1_000    # print interval

if __name__ == "__main__":
    cfg     = SystemConfig()
    gfn_cfg = GFlowNetConfig()
    print(cfg.summary())
    print(f"\nGFlowNetConfig: N={gfn_cfg.N}, K={gfn_cfg.K}, episodes={gfn_cfg.n_episodes}")