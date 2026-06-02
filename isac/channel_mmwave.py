import numpy as np

def steering_vector(angle_deg, N):
    """
    ULA steering vector with half-wavelength spacing.
    Reference point: center of array (as in Eq.7).

    Parameters:
    angle_deg: angle in degrees (-90 to 90)
    N: number of antennas

    Returns:
    steering vector of shape (N,)
    """

    theta_rad = np.deg2rad(angle_deg)
    sin_theta = np.sin(theta_rad)

    # Indices: -(N-1)/2, -(N-3)/2, ..., (N-1)/2
    indices = np.arange(N) - (N - 1) / 2

    # Steering vector elements: e^{-j * π * index * sinθ}
    steering = np.exp(-1j * np.pi * indices * sin_theta)

    return steering

# Testing the function

# if __name__ == "__main__":
#     N = 10 # number of antennas (even, as in paper)
#     angles = [-30, 0, 30] 

#     print("=" * 60)
#     print("Testing Steering Vector (Eq. 7)")
#     print("=" * 60)

#     for angle in angles:
#         a = steering_vector(angle, N)
#         print(f"\nAngle = {angle}°:")
#         print(f" Shape: {a.shape}")
#         print(f" First elements: {np.round(a[:5], 4)}")
#         print(f"  Magnitude of first element: {np.abs(a[0]):.4f} (should be 1)")
#         print(f"  Check |a(θ)|^2 = {np.sum(np.abs(a)**2):.2f} (should be {N})")
        
def los_channel_component(angle_deg, N_t):
    """
    LoS component from Eq. (2):
    h_{L,i}^{LoS} = √(N_t) * a_t(ω_{i,0})

    Parameters:
    angle_deg: LoS angle of departure (degrees)
    N_t: number of transmit antennas

    Returns:
    LoS channel vector of shape (N_t,)
    """
    
    a_t = steering_vector(angle_deg, N_t)
    h_los = np.sqrt(N_t) * a_t
    return h_los

# # Test the LoS component
# if __name__ == "__main__":
#     N_t = 10
#     cu_angles = [40, 10, -30]       

#     print("=" * 60) 
#     print("LoS Channel Component (Eq. 2)")
#     print("=" * 60)

#     for angle in cu_angles:
#         h_los = los_channel_component(angle, N_t)
#         print(f"\nCU at {angle}°:")
#         print(f"  Shape: {h_los.shape}")
#         print(f"  First 5 elements: {np.round(h_los[:5], 4)}")
#         print(f"  Power ||h_los||² = {np.sum(np.abs(h_los)**2):.2f}")
#         print(f"  Expected power: {N_t**2} (since ||a_t||² = {N_t} and multiplied by √(N_t))")

def nlos_channel_component(N_t, L_p, rng=None):
    """
    NLoS scattering component from Eq. (2)
    h_{S,i}^{NLoS} = sqrt(N_t/L_p) * sum_{l=1}^{L_p} c_{i,l} * a_t(omega_{i,l})

    Parameters:
    N_t : number of transmit antennas
    L_p : number of scattering paths
    rng : numpy random generator

    Returns:
    NLoS channel vector of shape (N_t,)
    """
    if rng is None:
        rng = np.random.default_rng()

    h_nlos = np.zeros(N_t, dtype=complex)
    for l in range(L_p):
        c_il = (rng.standard_normal() + 1j*rng.standard_normal()) / np.sqrt(2)
        omega_il = rng.uniform(-np.pi/2, np.pi/2)
        omega_il_deg = np.degrees(omega_il)
        a_t = steering_vector(omega_il_deg,N_t)
        h_nlos += c_il * a_t
    return np.sqrt(N_t/L_p) * h_nlos 

def generate_cu_channel(N_t, kappa, L_p, angle_los_deg, rng=None):
    """
    Generate one CU Rician channel vector from Eq. (2).
    
    Parameters:
        N_t         : number of transmit antennas
    kappa       : Rician K-factor (v_i in paper)
    L_p         : number of scattering paths
    angle_los_deg : LoS AOD in degrees
    rng         : numpy random generator
    
    Returns:
    h_i : Rician channel vector of shape (N_t,)
    """
    if rng is None:
        rng = np.random.default_rng()

    h_los = los_channel_component(angle_los_deg, N_t)
    h_nlos = nlos_channel_component(N_t, L_p, rng)

    h_i = np.sqrt(kappa/(1 + kappa)) * h_los + np.sqrt(1/(1 + kappa)) * h_nlos
    return h_i


# if __name__ == "__main__":
#     rng   = np.random.default_rng(42)
#     N_t   = 8
#     kappa = 10.0   # strong LoS
#     L_p   = 3      # scattering paths
#     angle = 30.0   # LoS AOD

#     h = generate_cu_channel(N_t, kappa, L_p, angle, rng)
#     print(f"Shape        : {h.shape}")
#     print(f"||h_i||²     : {np.sum(np.abs(h)**2):.4f}")
#     print(f"Expected ~   : {N_t**2:.4f}")

def generate_eve_channel(N_t, theta_e_deg, sigma_alpha, rng=None):
    """
    Eve downlink channel for mmWave ISAC (Eq. 9, CAML paper).
    g_e = alpha_k * b(theta_k)

    where alpha_k ~ CN(0, sigma_alpha^2) is the complex
    path-loss coefficient (CAML paper, text below Eq. 9)
    and b(theta_k) is the transmit steering vector (Eq. 7).

    Parameters: 
    N_t         : number of transmit antennas
    theta_e_deg : Eve direction in degrees
    sigma_alpha : std dev of complex path-loss coefficient
    rng         : numpy random generator

    Returns: 
    g_e : Eve channel vector of shape (N_t,)
    """
    if rng is None:
        rng = np.random.default_rng()
    alpha_k = (rng.standard_normal() + 1j*rng.standard_normal())/np.sqrt(2) * sigma_alpha
    b_theta_k = steering_vector(theta_e_deg, N_t)
    return alpha_k * b_theta_k

# if __name__ == "__main__":
#     rng  = np.random.default_rng(42)
#     N_t  = 8
#     theta_e_deg = 30.0
#     sigma_alpha = 1.0

#     g_e = generate_eve_channel(N_t, theta_e_deg, sigma_alpha, rng)
#     print(f"Shape        : {g_e.shape}")
#     print(f"||g_e||²     : {np.sum(np.abs(g_e)**2):.4f}")
#     print(f"Expected ~   : {N_t:.4f}  (sigma_alpha=1 → E[||g_e||²] = N_t)")

def generate_channels_mmwave(
    N_t,
    N,
    kappa,
    L_p,
    sigma_alpha,
    theta_E_min_deg,
    theta_E_max_deg,
    cu_angle_min_deg = -90.0,
    cu_angle_max_deg =  90.0,
    rng  = None,
    seed = None,
):
    """
    Generate one full mmWave Monte Carlo channel realization.

    CU channels  : Rician fading (Eq. 2, CAML paper)
                   h_i = sqrt(v/(1+v)) * h_LoS + sqrt(1/(1+v)) * h_NLoS
    Eve channel  : Geometric (CAML paper, text below Eq. 9)
                   g_e = alpha_k * b(theta_k),  alpha_k ~ CN(0, sigma_alpha^2)

    Parameters:
    N_t              : number of transmit antennas
    N                : number of CUs
    kappa            : Rician K-factor (v_i in paper)
    L_p              : number of NLoS scattering paths per CU
    sigma_alpha      : std dev of Eve complex path-loss coefficient
    theta_E_min_deg  : Eve angle min [degrees]
    theta_E_max_deg  : Eve angle max [degrees]
    cu_angle_min_deg : CU LoS angle min [degrees] (default -90)
    cu_angle_max_deg : CU LoS angle max [degrees] (default  90)
    rng              : numpy random generator
    seed             : random seed (used if rng is None)

    Returns:
    dict with keys:
        H       : (N_t, N) complex CU channel matrix
        g_e     : (N_t,)   complex Eve channel vector
        theta_E : float    Eve angle [degrees]
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    # 1. Eve angle — uniform over restricted sector
    theta_E = float(rng.uniform(theta_E_min_deg, theta_E_max_deg))

    # 2. CU channels — each with independent random LoS angle
    H = np.zeros((N_t, N), dtype=complex)
    for i in range(N):
        angle_los = float(rng.uniform(cu_angle_min_deg, cu_angle_max_deg))
        H[:, i]   = generate_cu_channel(N_t, kappa, L_p, angle_los, rng)

    # 3. Eve channel
    g_e = generate_eve_channel(N_t, theta_E, sigma_alpha, rng)

    return {
        "H"      : H,
        "g_e"    : g_e,
        "theta_E": theta_E,
    }

