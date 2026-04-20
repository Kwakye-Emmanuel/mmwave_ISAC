"""Signal processing for ISAC-aided physical layer security.

Implements ZF precoding, AN covariance design, SINR, and secrecy rate.

Three curves:
    B1 — Conventional BF    : ZF + isotropic AN + random scheduling
    B2 — Sensing-assisted   : ZF + directed AN  + random scheduling
    Proposed                : ZF + directed AN  + DL scheduling

Power allocation:
    rho     : fraction for data  (default 0.5)
    1-rho   : fraction for AN
    ||w_k||^2 = rho * P_t / Kd

References:
    [Paper]  eqs. (5)-(15), Section III-A
    [Su24]   N. Su et al., IEEE TWC, 2024.
    [Cao25]  Y. Cao et al., IEEE TWC, 2025.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# ZF precoder  (paper eq. 5)
# ---------------------------------------------------------------------------

def zf_precoder(
    H_D:  NDArray[np.complexfloating],
    P_t:  float,
    rho:  float,
    Kd:   int,
) -> NDArray[np.complexfloating]:
    """Zero-forcing precoder for Kd scheduled users (paper eq. 5).

    W_D = sqrt(rho*P_t) * H_D^H(H_D H_D^H)^{-1} * Pi

    where Pi normalises each column so ||w_k||^2 = rho*P_t/Kd.
    Property: h_k^H w_m = 0 for all k != m.

    Args:
        H_D : (Kd, M) scheduled user channel matrix  (rows = h_k^H)
        P_t : total transmit power [W]
        rho : power fraction for data transmission
        Kd  : number of scheduled users
    Returns:
        W_D : (M, Kd) normalised ZF precoding matrix
    """
    gram  = H_D @ H_D.conj().T
    W_ZF  = H_D.conj().T @ np.linalg.inv(gram)

    p_per_stream = np.sqrt(rho * P_t / Kd)
    for k in range(Kd):
        col_norm = np.linalg.norm(W_ZF[:, k])
        if col_norm > 1e-12:
            W_ZF[:, k] = W_ZF[:, k] / col_norm * p_per_stream

    return W_ZF


# ---------------------------------------------------------------------------
# Null-space projector  (paper eq. 6)
# ---------------------------------------------------------------------------

def null_space_projector(
    H_D: NDArray[np.complexfloating],
    M:   int,
) -> NDArray[np.complexfloating]:
    """Orthogonal complement projector of H_D (paper eq. 6).

    V = I_M - H_D^H (H_D H_D^H)^{-1} H_D

    Property: H_D @ V = 0  →  AN causes zero interference at CUs.

    Args:
        H_D : (Kd, M) scheduled user channel matrix
        M   : number of BS antennas
    Returns:
        V : (M, M) null-space projector
    """
    gram = H_D @ H_D.conj().T
    inv_gram_H = np.linalg.solve(gram, H_D)
    P_H  = H_D.conj().T @ inv_gram_H
    return np.eye(M) - P_H


# ---------------------------------------------------------------------------
# AN covariance — B2/Proposed: rank-1 directed  (paper eq. 8)
# ---------------------------------------------------------------------------
    """
    Returns:
        R_N : (M, M) AN covariance matrix
            R_N = Z Z^H where Z = sqrt(P_AN)*u*
            Equivalent to paper AN matrix Z
            for SINR computation via
            ||h^H Z||^2 = h^H R_N h
    """
def an_covariance_directed(
    H_D:     NDArray[np.complexfloating],
    M:       int,
    P_t:     float,
    rho:     float,
    g_hat_e: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Rank-1 directed AN covariance (paper eq. 8).

    u* = dominant eigenvector of V^H * g_hat_e * g_hat_e^H * V
    R_N = (1-rho)*P_t * u* * (u*)^H

    Concentrates all AN power toward estimated Eve direction
    within null(H_D). Requires M > Kd.

    Args:
        H_D     : (Kd, M) scheduled user channel matrix
        M       : BS antennas
        P_t     : total transmit power [W]
        rho     : power fraction for data
        g_hat_e : (M,) estimated Eve channel
    Returns:
        R_N : (M, M) rank-1 directed AN covariance
    """
    V    = null_space_projector(H_D, M)
    P_AN = (1.0 - rho) * P_t
    Vg   = V @ g_hat_e
    norm = np.linalg.norm(Vg)

    if norm < 1e-12:
        return an_covariance_isotropic(H_D, M, P_t, rho)

    u_star = Vg / norm
    return P_AN * np.outer(u_star, u_star.conj())


# ---------------------------------------------------------------------------
# AN covariance — B1: isotropic null-space
# ---------------------------------------------------------------------------

def an_covariance_isotropic(
    H_D:  NDArray[np.complexfloating],
    M:    int,
    P_t:  float,
    rho:  float,
) -> NDArray[np.complexfloating]:
    """Isotropic null-space AN covariance for B1 (paper Remark 1).

    R_N^iso = (1-rho)*P_t / (M-Kd) * V*V^H

    AN power spread uniformly across M-Kd null-space dimensions.
    Standard baseline when Eve direction is unknown.

    Args:
        H_D : (Kd, M) scheduled user channel matrix
        M   : BS antennas
        P_t : total transmit power [W]
        rho : power fraction for data
    Returns:
        R_N : (M, M) isotropic AN covariance
    """
    Kd   = H_D.shape[0]
    V    = null_space_projector(H_D, M)
    P_AN = (1.0 - rho) * P_t
    return (P_AN / (M - Kd)) * (V @ V.conj().T)


# ---------------------------------------------------------------------------
# SINR — user  (paper eq. 11)
# ---------------------------------------------------------------------------

def compute_sinr_user(
    h_k:      NDArray[np.complexfloating],
    W_D:      NDArray[np.complexfloating],
    R_N:      NDArray[np.complexfloating],
    k_idx:    int,
    sigma2_C: float,
) -> float:
    """User SINR (paper eq. 11).

    SINR_k = |h_k^H w_k|^2
             / (sum_{m!=k}|h_k^H w_m|^2 + h_k^H R_N h_k + sigma_C^2)

    With ZF and null-space AN:
        inter-user interference = 0
        h_k^H R_N h_k          = 0
    so SINR_k = |h_k^H w_k|^2 / sigma_C^2.
    """
    Kd      = W_D.shape[1]
    signal  = abs(h_k.conj() @ W_D[:, k_idx]) ** 2
    interf  = sum(
        abs(h_k.conj() @ W_D[:, m]) ** 2
        for m in range(Kd) if m != k_idx
    )
    an_leak = max(0.0, float(np.real(h_k.conj() @ R_N @ h_k)))
    return signal / (interf + an_leak + sigma2_C)


# ---------------------------------------------------------------------------
# SINR — Eve  (paper eq. 12)
# ---------------------------------------------------------------------------

def compute_sinr_eve(
    g_e:      NDArray[np.complexfloating],
    W_D:      NDArray[np.complexfloating],
    R_N:      NDArray[np.complexfloating],
    k_idx:    int,
    sigma2_C: float,
) -> float:
    """Eve SINR intercepting user k (paper eq. 12).

    SINR_E^(k) = |g_e^H w_k|^2
                 / (sum_{m!=k}|g_e^H w_m|^2 + g_e^H R_N g_e + sigma_C^2)

    Evaluated with TRUE g_e for performance assessment.
    Beamforming uses estimated g_hat_e from sensing stage.
    """
    Kd       = W_D.shape[1]
    signal   = abs(g_e.conj() @ W_D[:, k_idx]) ** 2
    interf   = sum(
        abs(g_e.conj() @ W_D[:, m]) ** 2
        for m in range(Kd) if m != k_idx
    )
    an_power = max(0.0, float(np.real(g_e.conj() @ R_N @ g_e)))
    return signal / (interf + an_power + sigma2_C)


# ---------------------------------------------------------------------------
# Secrecy rate  (paper eq. 13)
# ---------------------------------------------------------------------------

def compute_secrecy_rate(
    sinr_k:    float,
    sinr_e:    float,
    time_frac: float = 1.0,
) -> float:
    """Achievable secrecy rate (paper eq. 13).

    R_k^sec = (T_c/T) * [log2(1+SINR_k) - log2(1+SINR_E^(k))]^+
    """
    raw = np.log2(1.0 + sinr_k) - np.log2(1.0 + sinr_e)
    return time_frac * max(0.0, float(raw))


# ---------------------------------------------------------------------------
# Secrecy sum-rate  (paper eq. 14)
# ---------------------------------------------------------------------------

def compute_secrecy_sum_rate(
    H:         NDArray[np.complexfloating],
    g_e:       NDArray[np.complexfloating],
    sched_idx: list[int],
    g_hat_e:   NDArray[np.complexfloating] | None,
    P_t:       float,
    sigma2_C:  float,
    time_frac: float = 1.0,
    rho:       float = 0.5,
) -> float:
    """Secrecy sum-rate for a scheduling decision (paper eq. 14).

    Pipeline: ZF precoder → AN design → SINR → secrecy rate.

    g_hat_e = None  →  B1: isotropic AN  (no sensing)
    g_hat_e = (M,)  →  B2/Proposed: directed AN  (sensing)

    Args:
        H         : (M, N) full user channel matrix
        g_e       : (M,)   true Eve channel  [evaluation only]
        sched_idx : list of Kd scheduled user indices
        g_hat_e   : (M,) estimated Eve channel, or None for B1
        P_t       : total transmit power [W]
        sigma2_C  : communication noise variance [W]
        time_frac : T_c/T  (default 1.0)
        rho       : power split ratio  (default 0.5)
    Returns:
        R_sum^sec [bps/Hz]
    """
    M   = H.shape[0]
    Kd  = len(sched_idx)
    H_D = H[:, sched_idx].conj().T

    W_D = zf_precoder(H_D, P_t, rho, Kd)

    if g_hat_e is None:
        R_N = an_covariance_isotropic(H_D, M, P_t, rho)
    else:
        R_N = an_covariance_directed(H_D, M, P_t, rho, g_hat_e)

    R_sum = 0.0
    for k_idx, user_idx in enumerate(sched_idx):
        h_k    = H[:, user_idx]
        sinr_k = compute_sinr_user(h_k, W_D, R_N, k_idx, sigma2_C)
        sinr_e = compute_sinr_eve(g_e, W_D, R_N, k_idx, sigma2_C)
        R_sum += compute_secrecy_rate(sinr_k, sinr_e, time_frac)

    return R_sum