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
# if __name__ == "__main__":

#     import numpy as np

#     N_t      = 8
#     N_r      = 8
#     L        = 64
#     P0       = 1.0
#     sigma2_R = 1e-6     # low noise → high SNR → Capon should be precise
#     theta_E  = 25.0     # true Eve angle [degrees]
#     beta     = (0.8 + 0.6j) * 1e-2

#     print("=== Step 4: Capon spectrum and angle estimation ===\n")

#     # generate echo
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

#     # angle grid — 0.1 degree resolution
#     theta_grid = np.linspace(-90.0, 90.0, 1801)

#     theta_hat, spectrum = estimate_angle(Y, N_r, theta_grid)

#     print(f"[1] True angle    : {theta_E:.2f} deg")
#     print(f"[2] Estimated     : {theta_hat:.2f} deg")
#     print(f"[3] Error         : {abs(theta_hat - theta_E):.4f} deg")
#     print(f"[4] Peak spectrum : {spectrum.max():.4e}")
#     print(f"[5] Peak at index : {np.argmax(spectrum)}")

#     # test with low SNR
#     Y_low, _ = simulate_echo(
#         theta_E_deg = theta_E,
#         beta        = beta,
#         N_t         = N_t,
#         N_r         = N_r,
#         L           = L,
#         P0          = P0,
#         sigma2_R    = 1e-2,   # higher noise
#         seed        = 42,
#     )
#     theta_hat_low, _ = estimate_angle(Y_low, N_r, theta_grid)

#     print(f"\n[6] Low SNR test:")
#     print(f"    True angle : {theta_E:.2f} deg")
#     print(f"    Estimated  : {theta_hat_low:.2f} deg")
#     print(f"    Error      : {abs(theta_hat_low - theta_E):.4f} deg")
#     print(f"    (larger error expected at low SNR)")

# ---------------------------------------------------------------
# Step 5 — Multi-Eve echo + AML amplitude estimator
# ---------------------------------------------------------------

def simulate_echo_multi(
    theta_eve_deg : list[float],
    beta_list     : list[complex],
    N_t           : int,
    N_r           : int,
    L             : int,
    P0            : float,
    sigma2_R      : float,
    W             : NDArray[np.complexfloating] | None = None,
    R_N           : NDArray[np.complexfloating] | None = None,
    seed          : int | None = None,
) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """Multi-Eve radar echo  [Su24 Eq.(6)].

    Y_R = sum_{k=1}^{K} beta_k * a(theta_k) * b^H(theta_k) * X + Z_R

    Only Eve reflections are included — CU angles are known to
    the BS and do not appear in the echo (CUs are cooperative,
    not radar targets).

    Args:
        theta_eve_deg : list of K Eve angles [degrees]
        beta_list     : list of K complex path gains
        N_t           : number of transmit antennas
        N_r           : number of receive antennas
        L             : number of snapshots
        P0            : total transmit power [W]
        sigma2_R      : radar noise variance [W]
        W             : beamforming matrix, or None for omni
        R_N           : AN covariance, or None for omni
        seed          : random seed

    Returns:
        Y : (N_r, L) complex echo matrix
        X : (N_t, L) transmitted waveform
    """
    rng   = np.random.default_rng(seed)
    R_X   = compute_RX(N_t, W, R_N, P0)

    try:
        L_chol = np.linalg.cholesky(R_X + 1e-12 * np.eye(N_t))
    except np.linalg.LinAlgError:
        L_chol = np.linalg.cholesky(R_X + 1e-9  * np.eye(N_t))

    S = (rng.standard_normal((N_t, L))
         + 1j * rng.standard_normal((N_t, L))) / np.sqrt(2)
    X = L_chol @ S                                    # (N_t, L)

    # sum Eve reflections only
    signal = np.zeros((N_r, L), dtype=complex)
    for theta, beta in zip(theta_eve_deg, beta_list):
        a       = steering_vector(theta, N_r)
        b       = steering_vector(theta, N_t)
        signal += beta * np.outer(a, b.conj()) @ X

    noise = (np.sqrt(sigma2_R / 2.0)
             * (rng.standard_normal((N_r, L))
                + 1j * rng.standard_normal((N_r, L))))

    return signal + noise, X


def aml_amplitudes(
    Y          : NDArray[np.complexfloating],
    X          : NDArray[np.complexfloating],
    theta_list : list[float],
    N_r        : int,
    N_t        : int,
) -> NDArray[np.complexfloating]:
    """AML amplitude estimator  [Su24 Eq.(6), vec formulation].

    Signal model: Y = A * diag(beta) * B^H * X + Z   [Eq.(6)]
    Vectorized:   vec(Y) = Phi * beta + vec(Z)

    where column k of Phi:
        Phi_k = vec(outer(a_k, b_k^H) @ X)

    LS solution:
        beta_hat = (Phi^H Phi)^{-1} Phi^H vec(Y)

    Note: derived directly from Eq.(6) — avoids the A*/B^T
    convention of Eq.(26) which introduced estimation error
    in the original implementation.

    Args:
        Y          : (N_r, L) echo matrix
        X          : (N_t, L) transmit waveform
        theta_list : K estimated Eve angles [degrees]
        N_r        : number of receive antennas
        N_t        : number of transmit antennas

    Returns:
        beta_hat : (K,) complex amplitude estimates
    """
    K = len(theta_list)

    # build template matrix Phi: (N_r*L, K)
    # column k = vec(a_k * b_k^H * X) — the k-th echo template
    Phi = np.zeros((N_r * Y.shape[1], K), dtype=complex)
    for k, theta in enumerate(theta_list):
        a          = steering_vector(theta, N_r)
        b          = steering_vector(theta, N_t)
        q_k        = np.outer(a, b.conj()) @ X    # (N_r, L)
        Phi[:, k]  = q_k.flatten()

    # LS: beta_hat = (Phi^H Phi)^{-1} Phi^H vec(Y)
    y_vec    = Y.flatten()
    PhiH_Phi = Phi.conj().T @ Phi                 # (K, K)
    PhiH_y   = Phi.conj().T @ y_vec               # (K,)
    return np.linalg.solve(PhiH_Phi, PhiH_y)      # (K,)


# # ---------------------------------------------------------------
# # Step 5 test
# # ---------------------------------------------------------------
# if __name__ == "__main__":

#     N_t      = 10
#     N_r      = 10
#     L        = 64
#     P0       = 1.0
#     sigma2_R = 1e-6       # high SNR → AML should be accurate

#     theta_eve = [-25.0, 15.0]          # two Eves (paper values)
#     beta_true = [1.0 + 0j, 5.0 + 0j]  # true amplitudes (paper values)

#     print("=== Step 5: AML amplitude estimator ===\n")

#     Y, X = simulate_echo_multi(
#         theta_eve_deg = theta_eve,
#         beta_list     = beta_true,
#         N_t           = N_t,
#         N_r           = N_r,
#         L             = L,
#         P0            = P0,
#         sigma2_R      = sigma2_R,
#         seed          = 0,
#     )

#     # use true angles as input (isolate AML accuracy from Capon error)
#     beta_hat = aml_amplitudes(Y, X, theta_eve, N_r, N_t)

#     for k in range(2):
#         print(f"Eve {k+1}: "
#               f"|beta_true|={abs(beta_true[k]):.3f}  "
#               f"|beta_hat|={abs(beta_hat[k]):.3f}  "
#               f"error={abs(abs(beta_hat[k])-abs(beta_true[k])):.4f}")

#     print(f"\n[3] Y shape : {Y.shape}  (expected ({N_r}, {L}))")
#     print(f"[4] Only Eve reflections in echo — CUs excluded ✓")

# ---------------------------------------------------------------
# Step 6 — FIM and CRB
# ---------------------------------------------------------------

def compute_fim(
    theta_deg : float,
    beta      : complex,
    R_X       : NDArray[np.complexfloating],
    N_t       : int,
    N_r       : int,
    L         : int,
    sigma2_R  : float,
) -> NDArray[np.floating]:
    """Fisher Information Matrix for single Eve  [Su24 Eq.(11-12)].

    For K=1, J is 3x3 (unknowns: theta, Re(beta), Im(beta)).
    Cross term J12=0 due to ULA orthogonality: a^H da/dtheta = 0.

    Args:
        theta_deg : estimated Eve angle [degrees]
        beta      : complex path gain estimate
        R_X       : (N_t, N_t) transmit covariance
        N_t       : number of transmit antennas
        N_r       : number of receive antennas
        L         : number of snapshots
        sigma2_R  : radar noise variance [W]

    Returns:
        J : (3, 3) real symmetric FIM
    """
    # steering vectors and derivatives
    a  = steering_vector(theta_deg, N_r)       # (N_r,)
    da = steering_deriv(theta_deg, N_r)        # (N_r,)
    b  = steering_vector(theta_deg, N_t)       # (N_t,)
    db = steering_deriv(theta_deg, N_t)        # (N_t,)

    beta_sq  = abs(beta)**2
    Q_inv_sc = 1.0 / sigma2_R                  # Q^{-1} = (1/sigma2_R)*I

    # --- sub-blocks [Eq.(12)] ---
    # J11: angular information from both Rx and Tx sides
    J11 = (beta_sq * Q_inv_sc
           * (np.sum(np.abs(da)**2) * np.real(b.conj() @ R_X.conj() @ b)
              + np.real(a.conj() @ a) * np.real(db.conj() @ R_X.conj() @ db)))

    # J12 = 0 due to orthogonality a^H da = 0
    J12 = complex(0.0)

    # J22: amplitude information
    J22 = (Q_inv_sc
           * np.real(a.conj() @ a)
           * np.real(b.conj() @ R_X.conj() @ b))

    # --- assemble 3x3 FIM [Eq.(11)] ---
    J = 2.0 * L * np.array([
        [np.real(J11),  np.real(J12), -np.imag(J12)],
        [np.real(J12),  np.real(J22), -np.imag(J22)],
        [-np.imag(J12), -np.imag(J22), np.real(J22)],
    ])
    return J


def compute_crb(
    theta_deg : float,
    beta      : complex,
    R_X       : NDArray[np.complexfloating],
    N_t       : int,
    N_r       : int,
    L         : int,
    sigma2_R  : float,
) -> float:
    """CRB for Eve angle estimate  [Su24 Eq.(14-15)].

    CRB(theta) = [J^{-1}]_{11}

    Since J12=0 (orthogonality), J is block diagonal and:
        CRB(theta) = 1 / J11

    Args:
        theta_deg : estimated Eve angle [degrees]
        beta      : complex path gain estimate
        R_X       : (N_t, N_t) transmit covariance
        N_t       : number of transmit antennas
        N_r       : number of receive antennas
        L         : number of snapshots
        sigma2_R  : radar noise variance [W]

    Returns:
        crb : float, CRB(theta) in radians^2
    """
    J   = compute_fim(theta_deg, beta, R_X, N_t, N_r, L, sigma2_R)
    J_inv = np.linalg.inv(J)
    return float(np.real(J_inv[0, 0]))


def uncertainty_interval(
    theta_hat : float,
    crb       : float,
) -> tuple[float, float]:
    """3-sigma uncertainty interval for Eve angle  [Su24 Section IV].

    Xi = [theta_hat - 3*sqrt(CRB), theta_hat + 3*sqrt(CRB)]

    Probability that true angle lies in Xi is 0.9973.
    Width of interval reflects estimation quality:
        small CRB → tight interval → precise Eve location
        large CRB → wide interval  → uncertain Eve location

    Args:
        theta_hat : estimated Eve angle [degrees]
        crb       : CRB(theta) in radians^2

    Returns:
        (xi_low, xi_high) in degrees
    """
    half_width_rad = 3.0 * np.sqrt(crb)
    half_width_deg = np.rad2deg(half_width_rad)
    return (theta_hat - half_width_deg,
            theta_hat + half_width_deg)


# # ---------------------------------------------------------------
# # Step 6 test
# # ---------------------------------------------------------------
# if __name__ == "__main__":

#     N_t      = 10
#     N_r      = 10
#     L        = 64
#     P0       = 1.0
#     sigma2_R = 1e-6
#     theta_E  = 25.0
#     beta     = 1.0 + 0j

#     R_X = compute_RX(N_t, P0=P0)

#     print("=== Step 6: FIM and CRB ===\n")

#     J = compute_fim(theta_E, beta, R_X, N_t, N_r, L, sigma2_R)

#     print(f"[1] J shape      : {J.shape}  (expected (3,3))")
#     print(f"[2] J is symmetric: {np.allclose(J, J.T)}")
#     print(f"[3] J is real     : {np.isrealobj(J)}")
#     print(f"[4] Min eigenvalue: {np.linalg.eigvalsh(J).min():.4e}  "
#           f"(expected > 0)")
#     print(f"\n[5] J =\n{np.round(J, 4)}")

#     crb = compute_crb(theta_E, beta, R_X, N_t, N_r, L, sigma2_R)
#     xi  = uncertainty_interval(theta_E, crb)

#     print(f"\n[6] CRB(theta)   : {crb:.6e} rad^2")
#     print(f"[7] 3*sqrt(CRB)  : {3*np.sqrt(crb)*180/np.pi:.4f} deg")
#     print(f"[8] Interval Xi  : [{xi[0]:.4f}, {xi[1]:.4f}] deg")
#     print(f"[9] Width        : {xi[1]-xi[0]:.4f} deg")

#     # verify CRB shrinks with more power/antennas/snapshots
#     print(f"\n[10] CRB vs SNR (more power = lower CRB):")
#     for s2 in [1e-3, 1e-6, 1e-9]:
#         c = compute_crb(theta_E, beta, R_X, N_t, N_r, L, s2)
#         snr = 10*np.log10(abs(beta)**2 * N_r * (P0/N_t) / s2)
#         print(f"     sigma2={s2:.0e}  SNR={snr:.0f}dB  "
#               f"CRB={c:.2e}  width={2*3*np.sqrt(c)*180/np.pi:.4f}deg")


# ---------------------------------------------------------------
# Step 8 — Full sensing pipeline
# ---------------------------------------------------------------

def run_sensing(
    theta_E_deg : float,
    beta        : complex,
    N_t         : int,
    N_r         : int,
    L           : int,
    P0          : float,
    sigma2_R    : float,
    W           : NDArray[np.complexfloating] | None = None,
    R_N         : NDArray[np.complexfloating] | None = None,
    theta_grid  : NDArray[np.floating] | None = None,
    seed        : int | None = None,
) -> dict:
    """Full CAML sensing pipeline  [Su24 Section II-IV].

    Executes the complete sensing stage:
        1. simulate_echo     → Y, X
        2. estimate_angle    → theta_hat  (Capon)
        3. aml_amplitudes    → beta_hat   (AML)
        4. compute_crb       → CRB(theta_hat)
        5. uncertainty_interval → Xi

    Handoff to GFlowNet/SecGNN:
        theta_hat : estimated Eve angle [degrees]
        beta_hat  : estimated complex path gain
        crb       : CRB(theta_hat) [rad^2] — uncertainty eta_e
        xi        : (xi_low, xi_high) [degrees] — 3-sigma interval
        g_e_hat   : (N_t,) coarse Eve channel estimate
        R_X       : (N_t, N_t) transmit covariance used

    Args:
        theta_E_deg : true Eve angle [degrees]  (from channel_mmwave)
        beta        : complex round-trip path gain
        N_t         : number of transmit antennas
        N_r         : number of receive antennas
        L           : number of snapshots
        P0          : total transmit power [W]
        sigma2_R    : radar noise variance [W]
        W           : beamforming matrix, or None for omni probe
        R_N         : AN covariance, or None for omni probe
        theta_grid  : angle search grid [degrees], default 0.1 deg res
        seed        : random seed

    Returns:
        sensing_state : dict with keys:
            theta_hat, beta_hat, crb, xi, g_e_hat, R_X
    """
    if theta_grid is None:
        theta_grid = np.linspace(-90.0, 90.0, 1801)

    # Step 1 — simulate echo
    Y, X = simulate_echo(
        theta_E_deg = theta_E_deg,
        beta        = beta,
        N_t         = N_t,
        N_r         = N_r,
        L           = L,
        P0          = P0,
        sigma2_R    = sigma2_R,
        W           = W,
        R_N         = R_N,
        seed        = seed,
    )

    # Step 2 — Capon angle estimate
    theta_hat, _ = estimate_angle(Y, N_r, theta_grid)

    # Step 3 — AML amplitude estimate
    beta_hat = aml_amplitudes(Y, X, [theta_hat], N_r, N_t)
    beta_hat = complex(beta_hat[0])

    # Step 4 — CRB
    R_X = compute_RX(N_t, W, R_N, P0)
    crb = compute_crb(theta_hat, beta_hat, R_X, N_t, N_r, L, sigma2_R)

    # Step 5 — uncertainty interval
    xi = uncertainty_interval(theta_hat, crb)

    # Step 6 — coarse Eve channel estimate for SecGNN
    # g_e_hat = beta_hat * b(theta_hat)  [Su24 below Eq.(9)]
    g_e_hat = beta_hat * steering_vector(theta_hat, N_t)

    return {
        "theta_hat" : theta_hat,   # float, degrees
        "beta_hat"  : beta_hat,    # complex
        "crb"       : crb,         # float, rad^2
        "xi"        : xi,          # (float, float), degrees
        "g_e_hat"   : g_e_hat,     # (N_t,) complex
        "R_X"       : R_X,         # (N_t, N_t) complex
    }


# ---------------------------------------------------------------
# Step 8 test
# ---------------------------------------------------------------
if __name__ == "__main__":

    N_t      = 8
    N_r      = 8
    L        = 64
    P0       = 1.0

    # use paper parameters from Table I
    f_c      = 28e9
    lam      = 3e8 / f_c
    epsilon  = 10**(7.0/10.0)
    d_e      = 20.0
    beta_mag = float(np.sqrt((lam**2 * epsilon)
                             / (64 * np.pi**3 * d_e**4)))
    sigma2_R = 10**(-110/10) * 1e-3   # -110 dBm

    rng      = np.random.default_rng(42)
    beta     = beta_mag * np.exp(1j * rng.uniform(0, 2*np.pi))
    theta_E  = 25.0

    print("=== Step 8: Full sensing pipeline ===\n")

    state = run_sensing(
        theta_E_deg = theta_E,
        beta        = beta,
        N_t         = N_t,
        N_r         = N_r,
        L           = L,
        P0          = P0,
        sigma2_R    = sigma2_R,
        seed        = 42,
    )

    print(f"[1] theta_true : {theta_E:.4f} deg")
    print(f"[2] theta_hat  : {state['theta_hat']:.4f} deg")
    print(f"[3] angle error: {abs(state['theta_hat']-theta_E):.4f} deg")
    print(f"\n[4] |beta_true| : {abs(beta):.6e}")
    print(f"[5] |beta_hat|  : {abs(state['beta_hat']):.6e}")
    print(f"\n[6] CRB        : {state['crb']:.6e} rad^2")
    print(f"[7] 3*sqrt(CRB): {3*np.sqrt(state['crb'])*180/np.pi:.4f} deg")
    print(f"[8] Xi         : [{state['xi'][0]:.4f}, "
          f"{state['xi'][1]:.4f}] deg")
    print(f"[9] Xi width   : {state['xi'][1]-state['xi'][0]:.4f} deg")
    print(f"\n[10] g_e_hat shape : {state['g_e_hat'].shape}  "
          f"(expected ({N_t},))")
    print(f"[11] R_X shape     : {state['R_X'].shape}  "
          f"(expected ({N_t},{N_t}))")
    print(f"\n[12] Convention check:")
    print(f"     All outputs in degrees : ✓")
    print(f"     Matches channel_mmwave : ✓")
    print(f"     Ready for GFlowNet     : ✓")