import numpy as np


def steering_vector(angle_deg, N):
    theta_rad = np.deg2rad(angle_deg)
    indices   = np.arange(N) - (N - 1) / 2
    return np.exp(-1j * np.pi * indices * np.sin(theta_rad))


def los_channel_component(angle_deg, N_t):
    return np.sqrt(N_t) * steering_vector(angle_deg, N_t)


def nlos_channel_component(N_t, L_p, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    h_nlos = np.zeros(N_t, dtype=complex)
    for _ in range(L_p):
        c     = (rng.standard_normal() + 1j*rng.standard_normal()) / np.sqrt(2)
        omega = np.degrees(rng.uniform(-np.pi/2, np.pi/2))
        h_nlos += c * steering_vector(omega, N_t)
    return np.sqrt(N_t / L_p) * h_nlos


def generate_cu_channel(N_t, kappa, L_p, angle_los_deg, rng=None):
    """
    Rician CU channel (Su et al.) (TWC 2024) Eq. (2):

        h_i = sqrt(v/(1+v)) * h_los + sqrt(1/(1+v)) * h_nlos

    with
        h_los  = sqrt(N_t) * a(omega_0)                       (||h_los||^2  = N_t^2)
        h_nlos = sqrt(N_t/L_p) * sum_l c_l * a(omega_l)       (E[||h_nlos||^2] = N_t^2)


    Parameters:
        N_t          : number of transmit antennas
        kappa        : Rician K-factor (v_i)
        L_p          : number of NLoS paths
        angle_los_deg: LoS AOD [degrees]
        rng          : numpy random generator

    Returns:
        h_i : (N_t,) complex channel vector, E[||h_i||^2] = N_t^2
    """
    if rng is None:
        rng = np.random.default_rng()
    h_los  = los_channel_component(angle_los_deg, N_t)
    h_nlos = nlos_channel_component(N_t, L_p, rng)
    h_i    = (np.sqrt(kappa/(1+kappa)) * h_los +
              np.sqrt(1/(1+kappa))     * h_nlos)
    return h_i


def generate_eve_channel(N_t, theta_e_deg, sigma_alpha, rng=None):
    """
    Eve channel: g_e = alpha_k * b(theta_k)   (Su et al. Eq. 9).

    The complex path-loss coefficient alpha_k ~ CN(0, sigma_alpha^2) is the only
    distance-dependent quantity in Su et al.'s model. Per their Section VII (citing
    Yu et al. [46]), it is a zero-mean complex Gaussian whose variance follows the
    one-way inverse-square law:

        var(alpha_k) = sigma_alpha^2  proportional to  1 / d_e^2,

    where d_e is the BS-to-Eve distance. alpha_k is drawn ONCE per realization
    (constant over the observation interval / block fading).

    With ||b(theta)||^2 = N_t, E[||g_e||^2] = sigma_alpha^2 * N_t.

    Parameters:
        N_t         : number of transmit antennas
        theta_e_deg : Eve angle [degrees]
        sigma_alpha : Eve coefficient std dev (already includes 1/d_e scaling)
        rng         : numpy random generator

    Returns:
        g_e : (N_t,) complex Eve channel vector
    """
    if rng is None:
        rng = np.random.default_rng()
    alpha_k = (rng.standard_normal() + 1j*rng.standard_normal()) / np.sqrt(2) * sigma_alpha
    return alpha_k * steering_vector(theta_e_deg, N_t)


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
    d_0:      float  = 1.0,
    d_e:      float  = 1.0,
    rng  = None,
    seed = None,
):
    """
    Generate one mmWave channel realization (Su et al. TWC 2024 model).

    CU channels : normalized Rician (Eq. 2), E[||h_i||^2] = N_t^2, NO path loss.
    Eve channel : g_e = alpha_k * b(theta_e) (Eq. 9), alpha_k ~ CN(0, sigma_alpha^2 * sigma_e^2)
                  with one-way path loss sigma_e^2 = (d_0/d_e)^2  (variance ~ 1/d_e^2).

    Parameters:
        N_t              : transmit antennas
        N                : number of CUs
        kappa            : Rician K-factor (v_i)
        L_p              : NLoS paths per CU
        sigma_alpha      : Eve coefficient std dev (before distance scaling)
        theta_E_min_deg  : Eve angle min [degrees]
        theta_E_max_deg  : Eve angle max [degrees]
        cu_angle_min_deg : CU LoS angle min [degrees]
        cu_angle_max_deg : CU LoS angle max [degrees]
        d_0              : reference distance [m] for the Eve one-way path loss
        d_e              : BS-to-Eve distance [m]
        rng              : numpy random generator
        seed             : random seed

    Returns:
        dict: H (N_t, N), g_e (N_t,), theta_E (float)
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    theta_E = float(rng.uniform(theta_E_min_deg, theta_E_max_deg))

    # --- CU channels: normalized Rician, no path loss (Su et al. Eq. 2) ---
    H = np.zeros((N_t, N), dtype=complex)
    for i in range(N):
        angle_los = float(rng.uniform(cu_angle_min_deg, cu_angle_max_deg))
        H[:, i]   = generate_cu_channel(N_t, kappa, L_p, angle_los, rng)

    # --- Eve coefficient: one-way inverse-square path loss, var(alpha) ~ 1/d_e^2 ---
    sigma_e = (d_0 / d_e)            # amplitude; variance = (d_0/d_e)^2 ~ 1/d_e^2
    g_e     = generate_eve_channel(N_t, theta_E, sigma_alpha * sigma_e, rng)

    return {"H": H, "g_e": g_e, "theta_E": theta_E}


if __name__ == "__main__":

    N_t      = 10
    N        = 10
    kappa    = 0.1
    L_p      = 3
    sigma2_C = 1e-3
    sigma2_e = 1e-3
    phi      = 0.5
    K        = 2

    d_0      = 20.0     # reference distance for Eve one-way path loss
    d_e      = 20.0     # BS-to-Eve distance (Table I)

    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from beamforming import zf_precoder, an_covariance_isotropic, null_space_projector
    from signal_model import compute_sinr_cu, compute_sinr_eve, compute_secrecy_rate
    from itertools import combinations

    print("=" * 55)
    print("  channel_mmwave.py — Su et al. faithful (no CU path loss)")
    print("=" * 55)

    rng = np.random.default_rng(42)

    ch = generate_channels_mmwave(
        N_t=N_t, N=N, kappa=kappa, L_p=L_p,
        sigma_alpha=1.0,
        theta_E_min_deg=-30.0, theta_E_max_deg=30.0,
        d_0=d_0, d_e=d_e,
        seed=42,
    )

    norms_cu = np.array([np.sum(np.abs(ch["H"][:,k])**2) for k in range(N)])
    norm_eve = np.sum(np.abs(ch["g_e"])**2)

    print(f"\n[1] CU channel norms ||h_k||^2:")
    print(f"    mean={norms_cu.mean():.4f}  "
          f"min={norms_cu.min():.4f}  "
          f"max={norms_cu.max():.4f}")
    print(f"    Expected E[||h_k||^2] = N_t^2 = {N_t**2}  (no path loss)")

    print(f"\n[2] Eve channel norm ||g_e||^2:")
    print(f"    {norm_eve:.4f}")
    print(f"    E[||g_e||^2] = (sigma_alpha*d_0/d_e)^2 * N_t "
          f"= {(1.0*d_0/d_e)**2 * N_t:.2f}  (d_e=d_0 -> = N_t = {N_t})")

    print(f"\n[3] Secrecy rate vs P0 (oracle best subset, 200 trials):")
    print(f"    {'P0(dBm)':>8} {'P0(W)':>10} "
          f"{'SINR_CU(dB)':>13} {'R_sum':>8}")
    print("    " + "-"*43)

    for P0_dBm in [25, 30, 35]:
        P0   = 10**(P0_dBm/10) * 1e-3
        rng2 = np.random.default_rng(42)
        rates       = []
        sinr_cu_log = []

        for _ in range(200):
            ch2 = generate_channels_mmwave(
                N_t=N_t, N=N, kappa=kappa, L_p=L_p,
                sigma_alpha=1.0,
                theta_E_min_deg=-30.0, theta_E_max_deg=30.0,
                d_0=d_0, d_e=d_e,
                rng=rng2,
            )
            H2  = ch2["H"]
            g_e2 = ch2["g_e"]

            best = 0.0
            best_sc = 0.0
            for sel in combinations(range(N), K):
                H_K = H2[:, list(sel)]
                W   = zf_precoder(H_K, P0, phi, K)
                V   = null_space_projector(H_K, N_t)
                R_N = ((1-phi)*P0/(N_t-K)) * V
                sc  = [compute_sinr_cu(H2[:,sel[i]], W, R_N, i, sigma2_C)
                       for i in range(K)]
                se  = [compute_sinr_eve(g_e2, W, R_N, i, sigma2_e)
                       for i in range(K)]
                r   = compute_secrecy_rate(sc, se)
                if r > best:
                    best    = r
                    best_sc = float(np.mean(sc))
            rates.append(best)
            sinr_cu_log.append(best_sc)

        print(f"    {P0_dBm:>8} {P0:>10.4f} "
              f"{10*np.log10(np.mean(sinr_cu_log)):>13.1f} "
              f"{np.mean(rates):>8.3f}")

    print(f"\n    Note: with fixed ZF (max-SINR) precoding and no CU path loss,")
    print(f"    absolute R_sum sits ABOVE Su et al. Fig.6/8 (3-9 bits/s/Hz).")
    print(f"    Their range comes from joint W/R_N optimization, not path loss.")
    print(f"    Phase 1 reports relative ordering across schemes.")