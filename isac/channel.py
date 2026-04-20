from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from .sensing import ula_steering
from .config import SystemConfig

cfg = SystemConfig()

# ---------------------------------------------------------------------------
# Path loss
# ---------------------------------------------------------------------------
def compute_path_loss(
    d:   float | NDArray,
    d_0: float = cfg.d_0,
    eta: float = cfg.eta,
) -> float | NDArray:
    """Normalised path loss: beta_k = (d_0 / d_k)^eta."""
    return (d_0 / d) ** eta


# ---------------------------------------------------------------------------
# CU channel
# ---------------------------------------------------------------------------
def generate_channel_matrix(
    M:     int   = cfg.M,
    N:     int   = cfg.N,
    d_0:   float = cfg.d_0,
    eta:   float = cfg.eta,
    d_min: float = cfg.d_cu_min,
    d_max: float = cfg.d_cu_max,
    rng:   Generator | None = None,
) -> tuple[NDArray[np.complexfloating], NDArray[np.floating]]:
    """Generate Rayleigh fading CU channels.

    h_k = sqrt(beta_k) * h_tilde_k
    beta_k = (d_0/d_k)^eta
    d_k ~ Uniform[d_min, d_max]
    """
    if rng is None:
        rng = np.random.default_rng()

    d_users = rng.uniform(d_min, d_max, size=N)
    beta_k  = compute_path_loss(d_users, d_0, eta)

    H_small = (
        rng.standard_normal((M, N))
        + 1j * rng.standard_normal((M, N))
    ) / np.sqrt(2.0)

    H = H_small * np.sqrt(beta_k)[np.newaxis, :]
    return H, d_users


# ---------------------------------------------------------------------------
# Eve angle
# ---------------------------------------------------------------------------
def generate_eve_angle(
    theta_min: float = cfg.theta_E_min_rad,
    theta_max: float = cfg.theta_E_max_rad,
    rng:       Generator | None = None,
) -> float:
    """Sample Eve angle from Uniform[0°, 60°].

    - Avoids endfire (CRB degrades near ±90°)
    - Spatially separable from CU service region
    """
    if rng is None:
        rng = np.random.default_rng()
    return float(rng.uniform(theta_min, theta_max))


# ---------------------------------------------------------------------------
# Eve channel
# ---------------------------------------------------------------------------
def generate_eve_channel(
    M:          int   = cfg.M,
    theta_E:    float = 0.0,
    beta_e_mag: float = cfg.beta_e_mag,
    rng:        Generator | None = None,
) -> NDArray[np.complexfloating]:
    """Generate LoS Eve channel.

    g_e = beta_e * a(theta_E)
    |beta_e| = sqrt(eve_snr * sigma_C^2)
    Eve SNR = 20dB → |beta_e| = sqrt(100) = 10.0
    Random phase: beta_e = |beta_e| * exp(j*phi)
    """
    if rng is None:
        rng = np.random.default_rng()
    beta_e = beta_e_mag * np.exp(
        1j * rng.uniform(0.0, 2.0 * np.pi)
    )
    return beta_e * ula_steering(theta_E, M)


# ---------------------------------------------------------------------------
# Full channel realisation
# ---------------------------------------------------------------------------
def generate_channels(
    M:           int   = cfg.M,
    N:           int   = cfg.N,
    d_0:         float = cfg.d_0,
    d_be:        float = cfg.d_be,
    eta:         float = cfg.eta,
    d_cu_min:    float = cfg.d_cu_min,
    d_cu_max:    float = cfg.d_cu_max,
    beta_e_mag:  float = cfg.beta_e_mag,
    theta_E_min: float = cfg.theta_E_min_rad,
    theta_E_max: float = cfg.theta_E_max_rad,
    seed:        int | None = None,
) -> dict:
    """Generate one full Monte Carlo channel realisation."""
    rng     = np.random.default_rng(seed)
    theta_E = generate_eve_angle(theta_E_min, theta_E_max, rng)

    H, d_users = generate_channel_matrix(
        M, N,
        d_0   = d_0,
        eta   = eta,
        d_min = d_cu_min,
        d_max = d_cu_max,
        rng   = rng,
    )

    g_e = generate_eve_channel(M, theta_E, beta_e_mag, rng)

    return {
        "H":       H,
        "g_e":     g_e,
        "theta_E": theta_E,
        "d_users": d_users,
    }


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def generate_dataset(
    num_samples:  int   = cfg.num_samples,
    M:            int   = cfg.M,
    N:            int   = cfg.N,
    d_0:          float = cfg.d_0,
    d_be:         float = cfg.d_be,
    eta:          float = cfg.eta,
    d_cu_min:     float = cfg.d_cu_min,
    d_cu_max:     float = cfg.d_cu_max,
    beta_e_mag:   float = cfg.beta_e_mag,
    theta_E_min:  float = cfg.theta_E_min_rad,
    theta_E_max:  float = cfg.theta_E_max_rad,
    seed:         int   = cfg.seed,
) -> list[dict]:
    """Generate multiple channel realisations."""
    return [
        generate_channels(
            M           = M,
            N           = N,
            d_0         = d_0,
            d_be        = d_be,
            eta         = eta,
            d_cu_min    = d_cu_min,
            d_cu_max    = d_cu_max,
            beta_e_mag  = beta_e_mag,
            theta_E_min = theta_E_min,
            theta_E_max = theta_E_max,
            seed        = seed + i,
        )
        for i in range(num_samples)
    ]