"""sensing.py — ISAC sensing stage for physical layer security.

Implements the sensing pipeline from:
    [Su24] N. Su, F. Liu, C. Masouros, IEEE TWC, 2024.  (CAML paper)
    [Cao25] Y. Cao et al., IEEE TWC, 2025.  (closed-form CRB, single Eve)

"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


# Step 1 — Steering vector and its derivative

def steering_vector(angle_deg: float, N: int) -> NDArray[np.complexfloating]:
    """ULA steering vector — matches channel_mmwave.py exactly.

    a(theta) in C^N, half-wavelength spacing, center-referenced:
        a_n = exp(-j * pi * (n - (N-1)/2) * sin(theta))

    Properties guaranteed:
        |a_n| = 1  for all n          (purely imaginary exponent)
        ||a||^2 = N                   (constant modulus)
        a^H(theta) * da/dtheta = 0   (orthogonality — needed for FIM)

    Args:
        angle_deg : angle in degrees, in (-90, 90)
        N         : number of antennas

    Returns:
        a : (N,) complex steering vector
    """
    theta_rad = np.deg2rad(angle_deg)
    sin_theta = np.sin(theta_rad)
    indices   = np.arange(N) - (N - 1) / 2
    return np.exp(-1j * np.pi * indices * sin_theta)


def steering_deriv(angle_deg: float, N: int) -> NDArray[np.complexfloating]:
    """Derivative of ULA steering vector w.r.t. theta in RADIANS.

    da(theta)/d(theta_rad) = -j * pi * cos(theta) * Phi * a(theta)

    where Phi = diag(-(N-1)/2, ..., (N-1)/2)  [Su24 Eq.(11-12)]

    This is used in the FIM as Ȧ (receive) and Ḃ (transmit).

    Verified property:
        ||da/dtheta||^2 = pi^2 * cos^2(theta) * N*(N^2-1)/12

    Args:
        angle_deg : angle in degrees
        N         : number of antennas

    Returns:
        da : (N,) complex derivative vector  [units: 1/radian]
    """
    theta_rad = np.deg2rad(angle_deg)
    cos_theta = np.cos(theta_rad)
    indices   = np.arange(N) - (N - 1) / 2
    a         = steering_vector(angle_deg, N)
    return -1j * np.pi * cos_theta * indices * a


def steering_deriv_norm_sq(angle_deg: float, N: int) -> float:
    """Closed-form squared norm of steering derivative [Su24 Eq.(12)].

    ||da/dtheta||^2 = pi^2 * cos^2(theta) * N*(N^2-1) / 12

    Avoids computing the full derivative vector when only the
    norm is needed (e.g. closed-form CRB calculation).

    Args:
        angle_deg : angle in degrees
        N         : number of antennas

    Returns:
        float : ||da/dtheta||^2  [units: 1/radian^2]
    """
    theta_rad = np.deg2rad(angle_deg)
    return (np.pi * np.cos(theta_rad))**2 * N * (N**2 - 1) / 12.0



# # Step 1 test

# if __name__ == "__main__":
#     import numpy as np

#     N   = 8
#     ang = 30.0

#     a  = steering_vector(ang, N)
#     da = steering_deriv(ang, N)

#     print("=== Step 1: Steering vector and derivative ===\n")

#     print(f"[1] Constant modulus: all |a_n|=1 ? "
#           f"{np.allclose(np.abs(a), 1.0)}")

#     print(f"[2] ||a||^2 = {np.sum(np.abs(a)**2):.4f}  "
#           f"(expected {N})")

#     print(f"[3] Orthogonality a^H·da = "
#           f"{a.conj() @ da:.2e}  (expected 0)")

#     norm_numerical = np.sum(np.abs(da)**2)
#     norm_formula   = steering_deriv_norm_sq(ang, N)
#     print(f"[4] ||da||^2 numerical = {norm_numerical:.6f}")
#     print(f"    ||da||^2 formula   = {norm_formula:.6f}")
#     print(f"    Match: {np.isclose(norm_numerical, norm_formula)}")

#     print(f"\n[5] Convention check (must match channel_mmwave.py):")
#     print(f"    a[0]  = {a[0]:.6f}")
#     print(f"    a[-1] = {a[-1]:.6f}")
#     print(f"    (expected: 0.707107-0.707107j and 0.707107+0.707107j)")


# Step 2 — Transmit covariance matrix

def compute_RX(
    N_t : int,
    W   : NDArray[np.complexfloating] | None = None,
    R_N : NDArray[np.complexfloating] | None = None,
    P0  : float = 1.0,
) -> NDArray[np.complexfloating]:
    """Transmit covariance matrix R_X  [Su24 Eq.(13)].

    Two cases:
        Omnidirectional probe (W=None, R_N=None):
            R_X = (P0/N_t) * I_{N_t}
            Used at initialization — power spread equally
            in all directions for initial Eve detection.

        Beamforming case (W and R_N provided):
            R_X = W @ W^H + R_N
            Used after W and R_N are designed.

    Args:
        N_t : number of transmit antennas
        W   : (N_t, I) beamforming matrix, or None
        R_N : (N_t, N_t) AN covariance matrix, or None
        P0  : total transmit power [W]  (used only if W is None)

    Returns:
        R_X : (N_t, N_t) transmit covariance matrix
    """
    if W is None and R_N is None:
        # omnidirectional probe — Stage 0
        return (P0 / N_t) * np.eye(N_t, dtype=complex)

    if W is None or R_N is None:
        raise ValueError(
            "W and R_N must both be provided or both be None."
        )

    return W @ W.conj().T + R_N


# # Step 2 test

# if __name__ == "__main__":

#     N_t = 8
#     P0  = 1.0
#     I   = 3        # number of CUs

#     print("=== Step 2: Transmit covariance R_X ===\n")

#     # --- Case 1: omnidirectional probe ---
#     R_omni = compute_RX(N_t, W=None, R_N=None, P0=P0)

#     print("[Case 1] Omnidirectional probe:")
#     print(f"  Shape         : {R_omni.shape}")
#     print(f"  Is diagonal   : {np.allclose(R_omni, np.diag(np.diag(R_omni)))}")
#     print(f"  Diagonal value: {R_omni[0,0].real:.6f}  "
#           f"(expected {P0/N_t:.6f})")
#     print(f"  tr(R_X)       : {np.trace(R_omni).real:.6f}  "
#           f"(expected {P0:.6f})")

#     # --- Case 2: beamforming case ---
#     rng = np.random.default_rng(42)
#     W   = (rng.standard_normal((N_t, I))
#            + 1j * rng.standard_normal((N_t, I))) / np.sqrt(2)
#     R_N = np.eye(N_t) * 0.1

#     R_bf = compute_RX(N_t, W=W, R_N=R_N, P0=P0)

#     print(f"\n[Case 2] Beamforming case:")
#     print(f"  Shape            : {R_bf.shape}")
#     print(f"  Is Hermitian     : "
#           f"{np.allclose(R_bf, R_bf.conj().T)}")
#     print(f"  Is PSD (min eig) : "
#           f"{np.linalg.eigvalsh(R_bf).min():.6f}  (expected >= 0)")
#     print(f"  tr(R_X)          : {np.trace(R_bf).real:.6f}  "
#           f"(expected {np.trace(W@W.conj().T).real + 0.1*N_t:.6f})")

# Step 3 — Echo signal simulator

def simulate_echo(
    theta_E_deg : float,
    beta        : complex,
    N_t         : int,
    N_r         : int,
    L           : int,
    P0          : float,
    sigma2_R    : float,
    W           : NDArray[np.complexfloating] | None = None,
    R_N         : NDArray[np.complexfloating] | None = None,
    seed        : int | None = None,
) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """Simulate radar echo matrix Y_R  [Su24 Eq.(6)].

    Y_R = beta * a(theta_E) * b^H(theta_E) * X + Z_R

    where:
        a(theta_E) : (N_r,) receive steering vector
        b(theta_E) : (N_t,) transmit steering vector
        X          : (N_t, L) transmitted waveform
        Z_R        : (N_r, L) noise, columns ~ CN(0, sigma2_R * I)

    Waveform X is generated to satisfy:
        (1/L) * X @ X^H ≈ R_X = compute_RX(N_t, W, R_N, P0)

    Args:
        theta_E_deg : true Eve angle [degrees]
        beta        : complex round-trip path gain
        N_t         : number of transmit antennas
        N_r         : number of receive antennas
        L           : number of snapshots  (L >= N_t recommended)
        P0          : total transmit power [W]
        sigma2_R    : radar noise variance [W]
        W           : (N_t, I) beamforming matrix, or None for omni
        R_N         : (N_t, N_t) AN covariance, or None for omni
        seed        : random seed for reproducibility

    Returns:
        Y : (N_r, L) complex echo matrix
        X : (N_t, L) transmitted waveform  (kept for CAML estimator)
    """
    rng = np.random.default_rng(seed)

    # --- transmit waveform X satisfying (1/L) X X^H = R_X ---
    R_X = compute_RX(N_t, W, R_N, P0)

    # Cholesky factorization: R_X = L_chol @ L_chol^H
    # X = L_chol @ randn(N_t, L) * sqrt(L)
    # Then (1/L) X X^H → R_X as L → inf
    try:
        L_chol = np.linalg.cholesky(R_X + 1e-12 * np.eye(N_t))
    except np.linalg.LinAlgError:
        L_chol = np.linalg.cholesky(R_X + 1e-9 * np.eye(N_t))

    S = (rng.standard_normal((N_t, L))
         + 1j * rng.standard_normal((N_t, L))) / np.sqrt(2)
    X = L_chol @ S           # (N_t, L)

    # --- steering vectors ---
    a = steering_vector(theta_E_deg, N_r)         # (N_r,) receive
    b = steering_vector(theta_E_deg, N_t)         # (N_t,) transmit

    # --- echo signal ---
    # signal = beta * outer(a, b^H) @ X
    # shape:  (N_r,) * (N_r, N_t) @ (N_t, L) = (N_r, L)
    signal = beta * np.outer(a, b.conj()) @ X     # (N_r, L)

    # --- noise: each column ~ CN(0, sigma2_R * I_Nr) ---
    noise = (np.sqrt(sigma2_R / 2.0)
             * (rng.standard_normal((N_r, L))
                + 1j * rng.standard_normal((N_r, L))))

    return signal + noise, X


# # Step 3 test

# if __name__ == "__main__":

#     N_t      = 8
#     N_r      = 8
#     L        = 64
#     P0       = 1.0
#     sigma2_R = 1e-3
#     theta_E  = 25.0                          # degrees
#     beta     = (0.8 + 0.6j) * 1e-3          # complex path gain

#     print("=== Step 3: Echo simulator ===\n")

#     Y, X = simulate_echo(
#         theta_E_deg = theta_E,
#         beta        = beta,
#         N_t         = N_t,
#         N_r         = N_r,
#         L           = L,
#         P0          = P0,
#         sigma2_R    = sigma2_R,
#         seed        = 42,
#     )

#     print(f"[1] Y shape : {Y.shape}  (expected ({N_r}, {L}))")
#     print(f"[2] X shape : {X.shape}  (expected ({N_t}, {L}))")

#     # Check waveform covariance ≈ R_X = (P0/N_t)*I
#     R_X_empirical = (X @ X.conj().T) / L
#     R_X_expected  = (P0 / N_t) * np.eye(N_t)
#     print(f"\n[3] Waveform covariance check:")
#     print(f"    Diagonal mean  : {np.diag(R_X_empirical).real.mean():.6f}  "
#           f"(expected {P0/N_t:.6f})")
#     print(f"    Off-diag mean  : "
#           f"{np.abs(R_X_empirical - np.diag(np.diag(R_X_empirical))).mean():.6f}  "
#           f"(expected ~0, improves with larger L)")

#     # Check echo SNR — signal power vs noise power
#     a = steering_vector(theta_E, N_r)
#     b = steering_vector(theta_E, N_t)
#     signal_power = abs(beta)**2 * float(np.real(
#         b.conj() @ R_X_expected @ b
#     )) * N_r
#     noise_power  = sigma2_R
#     snr_dB       = 10 * np.log10(signal_power / noise_power)
#     print(f"\n[4] Echo SNR : {snr_dB:.2f} dB")
#     print(f"    |beta|^2  : {abs(beta)**2:.2e}")
#     print(f"    Signal power : {signal_power:.2e}")
#     print(f"    Noise power  : {noise_power:.2e}")

#     print(f"\n[5] Y stats:")
#     print(f"    Mean power per element : {np.mean(np.abs(Y)**2):.6e}")
#     print(f"    Max |Y_ij|             : {np.max(np.abs(Y)):.6e}")


# ---------------------------------------------------------------
# Step 4 — Capon spectrum and angle estimation
# ---------------------------------------------------------------

def capon_spectrum(
    Y         : NDArray[np.complexfloating],
    N_r       : int,
    theta_grid: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Capon spatial spectrum from echo matrix Y  [Su24 Section IV].

    P_Capon(theta) = 1 / (a^H(theta) * R_hat^{-1} * a(theta))

    where R_hat = (1/L) * Y @ Y^H  is the sample covariance
    of the received echo.

    The Capon spectrum has super-resolution — it can resolve
    closely-spaced sources better than a standard FFT beamformer.

    Args:
        Y          : (N_r, L) complex echo matrix
        N_r        : number of receive antennas
        theta_grid : (G,) angles to evaluate [degrees]

    Returns:
        spectrum : (G,) real Capon spectrum values
    """
    L = Y.shape[1]

    # sample covariance of echo
    R_hat = (Y @ Y.conj().T) / L                  # (N_r, N_r)

    # regularize for numerical stability
    R_hat = R_hat + 1e-10 * np.eye(N_r)

    # invert once — reuse for all angles
    R_inv = np.linalg.inv(R_hat)                  # (N_r, N_r)

    spectrum = np.zeros(len(theta_grid))
    for i, theta in enumerate(theta_grid):
        a          = steering_vector(theta, N_r)   # (N_r,)
        denom      = np.real(a.conj() @ R_inv @ a)
        spectrum[i] = 1.0 / denom

    return spectrum


def estimate_angle(
    Y          : NDArray[np.complexfloating],
    N_r        : int,
    theta_grid : NDArray[np.floating],
) -> tuple[float, NDArray[np.floating]]:
    """Estimate Eve's angle from echo using Capon  [Su24 Section IV].

    theta_hat = argmax_{theta} P_Capon(theta)

    Args:
        Y          : (N_r, L) complex echo matrix
        N_r        : number of receive antennas
        theta_grid : (G,) search grid [degrees]

    Returns:
        theta_hat : float, estimated angle [degrees]
        spectrum  : (G,) Capon spectrum (for inspection/plotting)
    """
    spectrum  = capon_spectrum(Y, N_r, theta_grid)
    theta_hat = float(theta_grid[np.argmax(spectrum)])
    return theta_hat, spectrum


# ---------------------------------------------------------------
# Step 4 test
# ---------------------------------------------------------------
if __name__ == "__main__":

    import numpy as np

    N_t      = 8
    N_r      = 8
    L        = 64
    P0       = 1.0
    sigma2_R = 1e-6     # low noise → high SNR → Capon should be precise
    theta_E  = 25.0     # true Eve angle [degrees]
    beta     = (0.8 + 0.6j) * 1e-2

    print("=== Step 4: Capon spectrum and angle estimation ===\n")

    # generate echo
    Y, X = simulate_echo(
        theta_E_deg = theta_E,
        beta        = beta,
        N_t         = N_t,
        N_r         = N_r,
        L           = L,
        P0          = P0,
        sigma2_R    = sigma2_R,
        seed        = 42,
    )

    # angle grid — 0.1 degree resolution
    theta_grid = np.linspace(-90.0, 90.0, 1801)

    theta_hat, spectrum = estimate_angle(Y, N_r, theta_grid)

    print(f"[1] True angle    : {theta_E:.2f} deg")
    print(f"[2] Estimated     : {theta_hat:.2f} deg")
    print(f"[3] Error         : {abs(theta_hat - theta_E):.4f} deg")
    print(f"[4] Peak spectrum : {spectrum.max():.4e}")
    print(f"[5] Peak at index : {np.argmax(spectrum)}")

    # test with low SNR
    Y_low, _ = simulate_echo(
        theta_E_deg = theta_E,
        beta        = beta,
        N_t         = N_t,
        N_r         = N_r,
        L           = L,
        P0          = P0,
        sigma2_R    = 1e-2,   # higher noise
        seed        = 42,
    )
    theta_hat_low, _ = estimate_angle(Y_low, N_r, theta_grid)

    print(f"\n[6] Low SNR test:")
    print(f"    True angle : {theta_E:.2f} deg")
    print(f"    Estimated  : {theta_hat_low:.2f} deg")
    print(f"    Error      : {abs(theta_hat_low - theta_E):.4f} deg")
    print(f"    (larger error expected at low SNR)")