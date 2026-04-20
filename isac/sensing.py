"""Sensing stage for ISAC-aided physical layer security.

Implements ULA steering, DFT codebook, echo simulation, MLE angle
estimation, LS path-gain estimation, and closed-form CRB for the
ISAC sensing stage.

Handoff to communication module:
    theta_hat : float    -- MLE angle estimate [rad]
    crb       : float    -- CRB(theta_E) [rad^2]
    at_hat    : NDArray  -- alpha_t(theta_hat), shape (M,)

Reference:
    [Cao25] Y. Cao et al., IEEE TWC, 2025.  Eqs. (5)-(16).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# ULA steering vector and derivative norm  (paper eq. 5, eq. 12)
# ---------------------------------------------------------------------------

def ula_steering(theta: float, N: int) -> NDArray[np.complexfloating]:
    """Centre-referenced ULA steering vector (paper eq. 5).

    alpha(theta) in C^{N}, half-wavelength spacing:
        alpha_n = exp(j * pi * sin(theta) * (n - (N-1)/2)),  n = 0,...,N-1

    Guaranteed properties:
        ||alpha||^2 = N
        alpha^H * d(alpha)/dtheta = 0  (cross-FIM terms vanish in CRB)

    Args:
        theta : angle [radians], theta in (-pi/2, pi/2)
        N     : number of antennas
    Returns:
        alpha : (N,) complex steering vector
    """
    n = np.arange(N) - (N - 1) / 2
    return np.exp(1j * np.pi * np.sin(theta) * n)


def ula_dot_norm_sq(theta: float, N: int) -> float:
    """Squared norm of ULA derivative vector, used in CRB (paper eq. 12).

    ||alpha_tilde||^2 = (pi * cos(theta))^2 * N * (N^2 - 1) / 12

    where alpha_tilde = d(alpha)/d(theta) = j*pi*cos(theta)*Phi*alpha,
    Phi = diag(-(N-1)/2, -(N-3)/2, ..., (N-1)/2).

    Returns:
        float
    """
    return (np.pi * np.cos(theta)) ** 2 * N * (N ** 2 - 1) / 12.0


# ---------------------------------------------------------------------------
# DFT sensing codebook  (paper eq. 6)
# ---------------------------------------------------------------------------

def dft_codebook(
    L: int,
    M: int,
) -> tuple[NDArray[np.complexfloating], NDArray[np.floating]]:
    """DFT beam codebook with L beams (paper eq. 6).

    Beam directions: theta_l = arcsin(-1 + (2l-1)/L),  l = 1,...,L
    Beam vectors:    v(l) = (1/sqrt(M)) * alpha(theta_l)

    Requires L >= M for R_X = (Ps/M)*I_{M} to hold (paper eq. 9).

    Args:
        L : number of sensing beams
        M : number of transmit antennas
    Returns:
        B      : (M, L) codebook matrix, unit-norm columns
        thetas : (L,) beam centre angles [radians]
    """
    l        = np.arange(1, L + 1)
    sin_vals = np.clip(-1.0 + (2.0 * l - 1.0) / L, -1 + 1e-9, 1 - 1e-9)
    thetas   = np.arcsin(sin_vals)
    B        = np.column_stack(
        [ula_steering(t, M) / np.sqrt(M) for t in thetas]
    )
    return B, thetas


# ---------------------------------------------------------------------------
# Radar path-gain magnitude  (paper eq. after eq. 7)
# ---------------------------------------------------------------------------

def compute_beta_s(
    d_be:         float,
    f_c:          float = 28e9,
    epsilon_dBsm: float = 7.0,
) -> float:
    """Round-trip path-gain magnitude from the radar equation.

    |beta_s|^2 = lambda^2 * epsilon / (64 * pi^3 * d_be^4)

    Random phase applied externally per trial.

    Args:
        d_be         : BS-Eve distance [m]
        f_c          : carrier frequency [Hz]   (default 28 GHz)
        epsilon_dBsm : radar cross section [dBsm] (default 7 dBsm)
    Returns:
        |beta_s| (float, positive)
    """
    lam     = 3e8 / f_c
    epsilon = 10 ** (epsilon_dBsm / 10.0)
    return float(
        np.sqrt((lam ** 2 * epsilon) / (64.0 * np.pi ** 3 * d_be ** 4))
    )


# ---------------------------------------------------------------------------
# Echo signal simulator  (paper eqs. 7-8)
# ---------------------------------------------------------------------------

def simulate_echo(
    theta_E:  float,
    M_t:      int,
    M_r:      int,
    L:        int,
    P_s:      float,
    sigma2_s: float,
    beta_s:   complex,
    seed:     int | None = None,
) -> NDArray[np.complexfloating]:
    """Simulate echo observation matrix Y (paper eq. 8).

    Y = beta_s * alpha_r(theta_E) * alpha_t^H(theta_E) * X + N

    where:
        X[:,l] = sqrt(P_s) * v(l) = sqrt(P_s/M_t) * alpha_t(theta_l)
        N[:,l] ~ CN(0, sigma_s^2 * I_{M_r})

    Args:
        theta_E  : true Eve angle [radians]
        M_t      : transmit antennas
        M_r      : receive antennas
        L        : number of sensing beams
        P_s      : sensing transmit power [W]
        sigma2_s : sensing noise variance [W]
        beta_s   : complex round-trip path gain
        seed     : random seed (None for unseeded)
    Returns:
        Y : (M_r, L) complex echo matrix
    """
    rng    = np.random.default_rng(seed)
    B, _   = dft_codebook(L, M_t)
    X      = np.sqrt(P_s) * B                               # (M_t, L)
    ar     = ula_steering(theta_E, M_r)                     # (M_r,)
    at     = ula_steering(theta_E, M_t)                     # (M_t,)
    signal = beta_s * np.outer(ar, at.conj()) @ X           # (M_r, L)
    noise  = np.sqrt(sigma2_s / 2.0) * (
        rng.standard_normal((M_r, L))
        + 1j * rng.standard_normal((M_r, L))
    )
    return signal + noise


# ---------------------------------------------------------------------------
# MLE angle estimator  (paper eq. 15)
# ---------------------------------------------------------------------------

def mle_estimate(
    Y:          NDArray[np.complexfloating],
    M_t:        int,
    M_r:        int,
    L:          int,
    P_s:        float,
    theta_grid: NDArray[np.floating],
) -> tuple[float, NDArray[np.floating]]:
    """MLE angle estimate for Eve's direction (paper eq. 15).

    theta_hat_E = argmax_{theta} |alpha_r^H(theta) * Y * X^H * alpha_t(theta)|^2

    For the DFT codebook with L >= M_t, the denominator
    ||X^H * alpha_t||^2 is constant, so the argmax of the
    numerator alone is the exact MLE.

    Args:
        Y          : (M_r, L) echo observation matrix
        M_t, M_r   : transmit / receive antennas
        L          : number of sensing beams
        P_s        : sensing transmit power [W]
        theta_grid : candidate angles to search [radians]
    Returns:
        theta_hat : MLE angle estimate [radians]
        spectrum  : raw objective values over theta_grid
    """
    B, _     = dft_codebook(L, M_t)
    X        = np.sqrt(P_s) * B                             # (M_t, L)
    spectrum = np.zeros(len(theta_grid))

    for i, th in enumerate(theta_grid):
        ar          = ula_steering(th, M_r)
        at          = ula_steering(th, M_t)
        u           = ar.conj() @ Y          # alpha_r^H Y,  shape (L,)
        v           = X.conj().T @ at        # X^H alpha_t,  shape (L,)
        spectrum[i] = abs(np.dot(u, v)) ** 2

    theta_hat = float(theta_grid[np.argmax(spectrum)])
    return theta_hat, spectrum


def mle_estimate_beta(
    Y:         NDArray[np.complexfloating],
    theta_hat: float,
    M_t:       int,
    M_r:       int,
    L:         int,
    P_s:       float,
) -> complex:
    """LS estimate of round-trip path gain beta_s (paper, after eq. 15).

    beta_hat_s = (vec(q(theta_hat)))^H * vec(Y) / ||vec(q(theta_hat))||^2

    where q(theta) = alpha_r(theta) * alpha_t^H(theta) * X.

    Args:
        Y         : (M_r, L) echo observation matrix
        theta_hat : MLE angle estimate [radians]
        M_t, M_r  : transmit / receive antennas
        L         : number of sensing beams
        P_s       : sensing transmit power [W]
    Returns:
        beta_hat_s : complex path-gain estimate
    """
    B, _  = dft_codebook(L, M_t)
    X     = np.sqrt(P_s) * B
    ar    = ula_steering(theta_hat, M_r)
    at    = ula_steering(theta_hat, M_t)
    Q     = np.outer(ar, at.conj()) @ X                     # (M_r, L)
    vQ    = Q.flatten()
    vY    = Y.flatten()
    return complex((vQ.conj() @ vY) / (vQ.conj() @ vQ))


# ---------------------------------------------------------------------------
# Cramér-Rao bound  (paper eq. 16, Theorem 1)
# ---------------------------------------------------------------------------

def crb_theta(
    theta_E:  float,
    M_t:      int,
    M_r:      int,
    L:        int,
    P_s:      float,
    sigma2_s: float,
    beta_mag: float,
) -> float:
    """Closed-form CRB for Eve's angle theta_E (paper eq. 16).

    CRB(theta_E) = sigma_s^2
                   / (2 * L * P_s * |beta_s|^2
                      * (||alpha_tilde_r||^2 + M_r/M_t * ||alpha_tilde_t||^2))

    Estimation error modelled as N(0, CRB(theta_E)):
        P(|theta_hat - theta_E| <= 3*sqrt(CRB)) = 0.9973.

    Args:
        theta_E  : true Eve angle [radians]
        M_t, M_r : transmit / receive antennas
        L        : number of DFT beams  (L >= M_t recommended)
        P_s      : sensing transmit power [W]
        sigma2_s : sensing noise variance [W]
        beta_mag : |beta_s|, round-trip path-gain magnitude
    Returns:
        CRB(theta_E) in radians^2
    """
    J = (
        2.0 * L * beta_mag ** 2 * P_s / sigma2_s
        * (
            ula_dot_norm_sq(theta_E, M_r)
            + M_r * ula_dot_norm_sq(theta_E, M_t) / M_t
        )
    )
    return 1.0 / J


# ---------------------------------------------------------------------------
# Sensing state  (output handed to the communication module)
# ---------------------------------------------------------------------------

def run_sensing(
    theta_E:  float,
    M_t:      int,
    M_r:      int,
    L:        int,
    P_s:      float,
    sigma2_s: float,
    beta_s:   complex,
    seed:     int | None = None,
) -> dict:
    """Run one sensing stage trial and return the sensing state.

    Executes: simulate_echo -> mle_estimate -> mle_estimate_beta -> crb_theta.

    The returned dict is the handoff to the communication module:
        theta_hat : MLE angle estimate [radians]
        crb       : CRB(theta_E) [radians^2]
        at_hat    : alpha_t(theta_hat), shape (M_t,)  -- for AN design
        beta_hat  : estimated round-trip path gain (complex)

    Args:
        theta_E  : true Eve angle [radians]
        M_t, M_r : transmit / receive antennas
        L        : number of sensing beams
        P_s      : sensing transmit power [W]
        sigma2_s : sensing noise variance [W]
        beta_s   : complex round-trip path gain (magnitude + random phase)
        seed     : random seed
    Returns:
        sensing_state : dict with keys theta_hat, crb, at_hat, beta_hat
    """
    theta_grid = np.linspace(-np.pi / 2, np.pi / 2, 1801)

    Y         = simulate_echo(theta_E, M_t, M_r, L, P_s, sigma2_s, beta_s, seed)
    theta_hat, _ = mle_estimate(Y, M_t, M_r, L, P_s, theta_grid)
    beta_hat  = mle_estimate_beta(Y, theta_hat, M_t, M_r, L, P_s)
    crb       = crb_theta(theta_E, M_t, M_r, L, P_s, sigma2_s, abs(beta_s))
    at_hat    = ula_steering(theta_hat, M_t)

    return {
        "theta_hat": theta_hat,
        "crb":       crb,
        "at_hat":    at_hat,
        "beta_hat":  beta_hat,
    }