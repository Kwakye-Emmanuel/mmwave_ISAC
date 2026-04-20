"""Central configuration for ISAC-aided physical layer security simulation.

All parameters match the paper's Table I and system model exactly.
Every other module imports from this file — no scattered magic numbers.

Usage:
    from config import SystemConfig
    cfg = SystemConfig()
    print(cfg.summary())
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SystemConfig:
    """Complete system and training configuration.

    Parameters match paper Table I exactly.
    """

    # ----------------------------------------------------------------
    # Antenna configuration
    # ----------------------------------------------------------------
    M:   int = 8    # BS transmit antennas
    M_r: int = 8    # BS receive antennas (M_r = M, symmetric ULA)

    # ----------------------------------------------------------------
    # User and scheduling
    # ----------------------------------------------------------------
    N:  int = 10   # total CUs
    Kd: int = 2    # scheduled users per frame

    # ----------------------------------------------------------------
    # Sensing stage  (Cao et al. TWC 2025, Table I)
    # ----------------------------------------------------------------
    L:            int   = 16       # sensing beams (L > M=8)
    P_s_dBm:      float = 20.0    # sensing transmit power [dBm]
    sigma2_dBm:   float = -110.0  # sensing noise power [dBm]
    f_c:          float = 28e9    # carrier frequency [Hz] (code only)
    epsilon_dBsm: float = 7.0     # radar cross section [dBsm]

    # ----------------------------------------------------------------
    # Channel model  (paper eq. 3, Table I)
    # ----------------------------------------------------------------
    eta:  float = 2.5    # path loss exponent

    # Path loss reference (separate from Eve distance):
    #   beta_k = (d_0 / d_k)^eta
    #   At d_k = d_0 = 30m: beta_k = 1.0
    d_0:  float = 20.0   # path loss reference distance [m]

    # CU distances: Uniform[d_cu_min, d_cu_max] from BS
    d_cu_min: float = 40.0   # min CU distance [m]
    d_cu_max: float = 60.0   # max CU distance [m]

    # Eve: physical distance and restricted angle sector
    d_be: float = 20.0   # Eve physical distance [m]
			 # Closer to BS than all CUs 
                                
    theta_E_min: float = 0.0  # Eve angle min [deg]
    theta_E_max: float = 60.0  # Eve angle max [deg]

    # ----------------------------------------------------------------
    # Communication stage  (paper Table I)
    # ----------------------------------------------------------------
    sigma2_C:   float = 1.0    # comm noise variance (normalised)
    rho:        float = 0.5    # power split ratio  (data / AN)
    eve_snr_dB: float = 20.0   # fixed Eve SNR [dB]

    # ----------------------------------------------------------------
    # Simulation SNR range  (paper Table I)
    # ----------------------------------------------------------------
    snr_min_dB: float = 0.0
    snr_max_dB: float = 20.0
    n_snr_pts:  int   = 9
    n_trials:   int   = 10000

    # ----------------------------------------------------------------
    # Secrecy outage threshold  (lab paper: R0 = 1 bps/Hz)
    # ----------------------------------------------------------------
    R0: float = 1.0   # target secrecy rate threshold [bps/Hz]

    # ----------------------------------------------------------------
    # Dataset generation  (paper: 10^6 training samples)
    # ----------------------------------------------------------------
    num_samples: int = 1_000_000
    seed:        int = 42

    # ----------------------------------------------------------------
    # Model architecture  (paper Table I)
    # ----------------------------------------------------------------
    embed_dim:  int   = 128
    num_heads:  int   = 4
    num_layers: int   = 2
    ff_dim:     int   = 256
    dropout:    float = 0.1

    # Cardinality regularizer weight
    card_weight: float = 0.1

    # ----------------------------------------------------------------
    # Training  (paper Table I)
    # ----------------------------------------------------------------
    num_epochs:    int   = 50
    learning_rate: float = 1e-3
    weight_decay:  float = 1e-4
    batch_size:    int   = 256
    val_split:     float = 0.1
    device:        str   = "auto"

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
    def local_dim(self) -> int:
        """Per-user feature: [Re(h_k), Im(h_k)] in R^{2M}."""
        return 2 * self.M

    @property
    def global_dim(self) -> int:
        """Global feature: [theta_hat, CRB, mu, sigma_h^2, rho, SNR]."""
        return 6

    @property
    def time_frac(self) -> float:
        """Communication time fraction (T_c/T = 1.0)."""
        return 1.0

    @property
    def P_s(self) -> float:
        """Sensing transmit power [W]."""
        return 10 ** ((self.P_s_dBm - 30.0) / 10.0)

    @property
    def sigma2_s(self) -> float:
        """Sensing noise variance [W]."""
        return 10 ** ((self.sigma2_dBm - 30.0) / 10.0)

    @property
    def eve_snr_linear(self) -> float:
        """Eve SNR linear."""
        return 10 ** (self.eve_snr_dB / 10.0)

    @property
    def beta_e_mag(self) -> float:
        """|beta_e| = sqrt(eve_snr * sigma_C^2)."""
        import math
        return math.sqrt(self.eve_snr_linear * self.sigma2_C)

    @property
    def theta_E_min_rad(self) -> float:
        """Eve min angle [rad]."""
        import math
        return math.radians(self.theta_E_min)

    @property
    def theta_E_max_rad(self) -> float:
        """Eve max angle [rad]."""
        import math
        return math.radians(self.theta_E_max)

    def summary(self) -> str:
        """Print-friendly parameter summary."""
        lines = [
            "=" * 60,
            "  SystemConfig  (paper Table I)",
            "=" * 60,
            f"  Antennas   : M={self.M}, M_r={self.M_r}",
            f"  Users      : N={self.N}, Kd={self.Kd}",
            f"  Sensing    : L={self.L}, "
            f"P_s={self.P_s_dBm} dBm, "
            f"sigma2={self.sigma2_dBm} dBm",
            f"               epsilon={self.epsilon_dBsm} dBsm",
            f"  Channel    : eta={self.eta}",
            f"               beta_k=(d_0/d_k)^eta, "
            f"d_0={self.d_0}m",
            f"               d_CU ~ Uniform[{self.d_cu_min}, "
            f"{self.d_cu_max}]m",
            f"               d_Eve = {self.d_be}m  "
            f"(outside CU zone)",
            f"               theta_E ~ Uniform["
            f"{self.theta_E_min}°, {self.theta_E_max}°]",
            f"  Comm       : rho={self.rho}, "
            f"sigma2_C={self.sigma2_C}",
            f"               Eve SNR = {self.eve_snr_dB} dB",
            f"  Outage     : R0 = {self.R0} bps/Hz",
            f"  SNR range  : [{self.snr_min_dB}, "
            f"{self.snr_max_dB}] dB "
            f"({self.n_snr_pts} pts)",
            f"  Features   : local_dim={self.local_dim}, "
            f"global_dim={self.global_dim}",
            f"  Model      : embed={self.embed_dim}, "
            f"heads={self.num_heads}, "
            f"layers={self.num_layers}",
            f"  Training   : epochs={self.num_epochs}, "
            f"lr={self.learning_rate}, "
            f"bs={self.batch_size}",
            f"  Samples    : {self.num_samples:,}",
            f"  Trials     : {self.n_trials}",
            "=" * 60,
        ]
        return "\n".join(lines)
