"""Beamforming module for mmWave ISAC physical layer security.

Implements ZF precoding and null-space AN beamforming.

References:
    [Paper] eqs. (10)-(12)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def zf_precoder(
    H_K:  NDArray[np.complexfloating],
    P_t:  float,
    phi:  float,
    K:    int,
) -> NDArray[np.complexfloating]:
    """Zero-forcing precoder for K scheduled users (paper Eq. 10).

    W = sqrt(phi * P_t) * H_K (H_K^H H_K)^{-1} * Pi

    where Pi normalises each column so ||w_k||^2 = phi * P_t / K.

    Parameters:
    H_K : (N_t, K) stacked channel matrix of scheduled users
    P_t : total transmit power [W]
    phi : power split ratio for data
    K   : number of scheduled users

    Returns:
    W : (N_t, K) normalised ZF precoding matrix
    """
    # Gram matrix (K, K)
    gram = H_K.conj().T @ H_K

    # Pseudo-inverse: H_K (H_K^H H_K)^{-1} → (N_t, K)
    W_ZF = H_K @ np.linalg.inv(gram)

    # Normalize each column so ||w_k||^2 = phi * P_t / K
    p_per_stream = np.sqrt(phi * P_t / K)
    for k in range(K):
        col_norm = np.linalg.norm(W_ZF[:, k])
        if col_norm > 1e-12:
            W_ZF[:, k] = W_ZF[:, k] / col_norm * p_per_stream

    return W_ZF



def null_space_projector(
    H_K: NDArray[np.complexfloating],
    N_t: int,
) -> NDArray[np.complexfloating]:
    """Null-space projector of H_K^H (paper Eq. 12).

    V = I_M - H_K (H_K^H H_K)^{-1} H_K^H

    Property: H_K^H V = 0 — AN causes zero interference at CUs.
    Requires N_t > K.

    Parameters:
    H_K : (N_t, K) stacked channel matrix of scheduled users
    N_t : number of transmit antennas

    Returns:
    V : (N_t, N_t) null-space projector
    """
    gram    = H_K.conj().T @ H_K                        # (K, K)
    P_H     = H_K @ np.linalg.inv(gram) @ H_K.conj().T # (N_t, N_t)
    return np.eye(N_t) - P_H


def an_covariance_isotropic(
    H_K: NDArray[np.complexfloating],
    N_t: int,
    P_t: float,
    phi: float,
    K:   int,
) -> NDArray[np.complexfloating]:
    """Isotropic null-space AN covariance (paper Eq. 11).

    R_N = (1-phi)*P_t / (N_t-K) * V

    AN power spread uniformly across N_t-K null-space dimensions.
    Used when Eve direction is unknown.

    Parameters:
    H_K : (N_t, K) stacked channel matrix of scheduled users
    N_t : number of transmit antennas
    P_t : total transmit power [W]
    phi : power split ratio for data
    K   : number of scheduled users

    Returns:
    R_N : (N_t, N_t) isotropic AN covariance matrix
    """
    V     = null_space_projector(H_K, N_t)
    P_AN  = (1.0 - phi) * P_t
    return (P_AN / (N_t - K)) * V




def an_covariance_directed(
    H_K    : NDArray[np.complexfloating],
    g_e_hat: NDArray[np.complexfloating],
    N_t    : int,
    P_t    : float,
    phi    : float,
    K      : int,
) -> NDArray[np.complexfloating]:
    """Rank-1 directed AN covariance toward estimated Eve direction.

    R_N = P_AN * u* (u*)^H
    where u* = V @ g_e_hat / ||V @ g_e_hat||
          V  = null_space_projector(H_K)

    Concentrates all AN power toward Eve's estimated direction
    within the null space of H_K — zero interference at CUs.

    Properties:
        H_K^H R_N ≈ 0       zero AN leakage at CUs
        tr(R_N) = (1-phi)*P_t
        rank(R_N) = 1        all AN toward Eve

    Args:
        H_K     : (N_t, K) scheduled CU channel matrix
        g_e_hat : (N_t,)   estimated Eve channel from sensing
        N_t     : number of transmit antennas
        P_t     : total transmit power [W]
        phi     : power split ratio (data fraction)
        K       : number of scheduled users

    Returns:
        R_N : (N_t, N_t) rank-1 directed AN covariance
    """
    V    = null_space_projector(H_K, N_t)
    P_AN = (1.0 - phi) * P_t
    Vg   = V @ g_e_hat
    norm = np.linalg.norm(Vg)
    if norm < 1e-12:
        # fallback to isotropic if Eve direction is in signal space
        return an_covariance_isotropic(H_K, N_t, P_t, phi, K)
    u_star = Vg / norm
    return P_AN * np.outer(u_star, u_star.conj())

if __name__ == "__main__":
    import numpy as np
    from channel_mmwave import generate_channels_mmwave

    N_t = 10
    K   = 2
    P_t = 1.0
    phi = 0.5

    # Generate channel
    ch  = generate_channels_mmwave(
        N_t=N_t, N=10, kappa=0.1, L_p=3,
        sigma_alpha=1.0,
        theta_E_min_deg=-30.0,
        theta_E_max_deg= 30.0,
        seed=42,
    )

    # Pick first K users
    sched_idx = [0, 1]
    H_K = ch["H"][:, sched_idx]   # (N_t, K)

    W = zf_precoder(H_K, P_t, phi, K)

    print(f"W shape        : {W.shape}")
    print(f"||w_0||^2      : {np.linalg.norm(W[:, 0])**2:.4f}")
    print(f"||w_1||^2      : {np.linalg.norm(W[:, 1])**2:.4f}")
    print(f"Expected       : {phi * P_t / K:.4f}")

    # ZF property: h_k^H w_m = 0 for k != m
    h_0 = H_K[:, 0]
    h_1 = H_K[:, 1]
    print(f"\nZF check:")
    print(f"|h_0^H w_1|    : {abs(h_0.conj() @ W[:, 1]):.6f} (should be ~0)")
    print(f"|h_1^H w_0|    : {abs(h_1.conj() @ W[:, 0]):.6f} (should be ~0)")

    V = null_space_projector(H_K, N_t)

    print(f"\nV shape        : {V.shape}")
    print(f"H_K^H V ~ 0    : {np.max(np.abs(H_K.conj().T @ V)):.6f} (should be ~0)")
    print(f"V is projector : {np.max(np.abs(V @ V - V)):.6f} (should be ~0)")

    # --- Isotropic AN Covariance ---
    R_N = an_covariance_isotropic(H_K, N_t, P_t, phi, K)

    print(f"\nR_N shape      : {R_N.shape}")
    print(f"tr(R_N)        : {np.real(np.trace(R_N)):.4f}")
    print(f"Expected tr    : {(1-phi)*P_t:.4f}")
    print(f"AN at CU 0     : {np.real(H_K[:,0].conj() @ R_N @ H_K[:,0]):.6f} (should be ~0)")
    print(f"AN at CU 1     : {np.real(H_K[:,1].conj() @ R_N @ H_K[:,1]):.6f} (should be ~0)")

