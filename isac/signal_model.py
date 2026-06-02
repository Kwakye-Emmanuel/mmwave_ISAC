from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

def generate_data_streams(I, L, rng=None):
    """
    Generate data stream matrix S of shape (I, L).
    Each entry ~ CN(0, 1), i.e., unit-power streams.

    Parameters:
    I   : number of CUs (streams)
    L   : number of time snapshots
    rng : numpy random generator

    Returns:
    S : (I, L) complex data stream matrix
    """
    if rng is None:
        rng = np.random.default_rng()

    real = rng.standard_normal((I, L))
    imag = rng.standard_normal((I, L))
    S = (real + 1j * imag) / np.sqrt(2)

    return S

def generate_an_matrix(R_N, L, rng=None):
    """
    Generate AN matrix N of shape (N_t, L).
    Each column ~ CN(0, R_N).

    Parameters:
    R_N : (N_t, N_t) AN covariance matrix
    L   : number of time snapshots
    rng : numpy random generator

    Returns:
    N : (N_t, L) complex AN matrix
    """
    if rng is None:
        rng = np.random.default_rng()

    N_t = R_N.shape[0]

    # Cholesky decomposition: R_N = L_chol @ L_chol^H
    L_chol = np.linalg.cholesky(R_N + 1e-12 * np.eye(N_t))

    # Draw standard CN(0, I) samples
    noise = (rng.standard_normal((N_t, L)) + 1j * rng.standard_normal((N_t, L))) / np.sqrt(2)

    # Color the noise: each column ~ CN(0, R_N)
    return L_chol @ noise

def generate_transmit_signal(W, S, N):
    """
    Generate transmit signal matrix X = WS + N.

    Parameters:
    W : (N_t, I)  beamforming matrix
    S : (I, L)    data stream matrix
    N : (N_t, L)  AN matrix

    Returns:
    X : (N_t, L) transmit signal matrix
    """
    return W @ S + N

def received_signal_cu(H, X, sigma2_C, rng=None):
    """
    Received signal at all CUs (paper Eq. 1).
    Y_C = H * X + Z_C

    Parameters:
    H       : (I, N_t)  full CU channel matrix (rows = h_i^H)
    X       : (N_t, L)  transmit signal matrix
    sigma2_C: noise variance at each CU
    rng     : numpy random generator

    Returns:
    Y_C : (I, L) received signal matrix
    """
    if rng is None:
        rng = np.random.default_rng()

    I, L  = H.shape[0], X.shape[1]
    Z_C   = (rng.standard_normal((I, L)) + 1j * rng.standard_normal((I, L))) / np.sqrt(2) * np.sqrt(sigma2_C)

    return H @ X + Z_C


def compute_sinr_cu(h_i, W, R_N, i, sigma2_C):
    """
    CU SINR for user i (paper Eq. 5).

    SINR_i^CU = |h_i^H w_i|²
                / (Σ_{m≠i} |h_i^H w_m|² + h_i^H R_N h_i + σ_C²)

    Parameters:
    h_i     : (N_t,)    channel vector of user i
    W       : (N_t, I)  beamforming matrix
    R_N     : (N_t, N_t) AN covariance matrix
    i       : index of user i in W
    sigma2_C: noise variance at CU

    Returns:
    sinr : float, SINR of user i
    """
    I = W.shape[1]

    signal = abs(h_i.conj() @ W[:, i]) ** 2

    interference = sum(
        abs(h_i.conj() @ W[:, m]) ** 2
        for m in range(I) if m != i
    )

    an_leakage = float(np.real(h_i.conj() @ R_N @ h_i))

    return signal / (interference + an_leakage + sigma2_C)

def compute_sinr_eve(g_e, W, R_N, i, sigma2_e):
    """
    Eve SINR intercepting user i (paper Eq. 9).

    SINR_{k,i}^E = g_e^H W_tilde_i g_e
                   / (g_e^H (sum_{m!=i} W_tilde_m + R_N) g_e + sigma_e^2)

    Parameters:
    g_e     : (N_t,)    Eve channel vector
    W       : (N_t, I)  beamforming matrix
    R_N     : (N_t, N_t) AN covariance matrix
    i       : index of target user stream
    sigma2_e: noise variance at Eve

    Returns:
    sinr : float, Eve SINR for user i's stream
    """
    I = W.shape[1]

    signal = float(np.real(g_e.conj() @ (W[:, i, None] @ W[:, i, None].conj().T) @ g_e))

    interference_matrix = sum(
        W[:, m, None] @ W[:, m, None].conj().T
        for m in range(I) if m != i
    ) + R_N

    denominator = float(np.real(g_e.conj() @ interference_matrix @ g_e)) + sigma2_e

    return signal / denominator

def compute_rate_cu(sinr_k):
    """
    Achievable rate of CU i (paper Eq. 17a).
    R_i^CU = log2(1 + SINR_i^CU)

    Parameters:
    sinr_k : float, CU SINR

    Returns:
    rate : float [bits/s/Hz]
    """
    return float(np.log2(1.0 + sinr_k))


def compute_rate_eve(sinr_e):
    """
    Achievable rate of Eve k for user i (paper Eq. 17b).
    R_{k,i}^E = log2(1 + SINR_{k,i}^E)

    Parameters:
    sinr_e : float, Eve SINR

    Returns:
    rate : float [bits/s/Hz]
    """
    return float(np.log2(1.0 + sinr_e))


def compute_secrecy_rate(sinr_cu, sinr_eve):
    """
    Secrecy sum-rate (updated to sum formulation).
    SR = Σ_k [log2(1 + SINR_CU_k) - log2(1 + SINR_Eve_k)]^+

    Parameters:
    sinr_cu  : list of CU SINRs  for all i
    sinr_eve : list of Eve SINRs for all k, i

    Returns:
    sr : float [bits/s/Hz]
    """
    sr = sum(
        max(0.0, compute_rate_cu(s_cu) - compute_rate_eve(s_eve))
        for s_cu, s_eve in zip(sinr_cu, sinr_eve)
    )
    return float(sr)
