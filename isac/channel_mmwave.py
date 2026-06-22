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


def generate_cu_channel(N_t, kappa, L_p, angle_los_deg, sigma_cu=1.0, rng=None):
    """
    Rician CU channel with path loss (Su et al. Eq.2).

    h_i = sigma_cu / sqrt(N_t) * (
              sqrt(v/(1+v)) * sqrt(N_t) * a(omega_0)
            + sqrt(1/(1+v)) * sqrt(N_t/L_p) * sum c_l * a(omega_l)
          )

    Normalization by sqrt(N_t) gives E[||h_i||²] = sigma_cu² * N_t.
    Path loss: sigma_cu = (d_0/d_k)^(alpha/2).

    Parameters:
        N_t          : number of transmit antennas
        kappa        : Rician K-factor
        L_p          : number of NLoS paths
        angle_los_deg: LoS AOD [degrees]
        sigma_cu     : path loss amplitude = (d_0/d_k)^(alpha/2)
        rng          : numpy random generator

    Returns:
        h_i : (N_t,) complex channel vector
    """
    if rng is None:
        rng = np.random.default_rng()
    h_los  = los_channel_component(angle_los_deg, N_t)
    h_nlos = nlos_channel_component(N_t, L_p, rng)
    h_i    = (np.sqrt(kappa/(1+kappa)) * h_los +
              np.sqrt(1/(1+kappa))     * h_nlos)
    return sigma_cu * h_i / np.sqrt(N_t)


def generate_eve_channel(N_t, theta_e_deg, sigma_alpha, rng=None):
    """
    Eve channel: g_e = alpha_k * b(theta_k)  (Su et al. Eq.9).

    alpha_k ~ CN(0, sigma_alpha²).
    sigma_alpha incorporates path loss: sigma_alpha * (d_0/d_e)^(alpha/2).

    Parameters:
        N_t         : number of transmit antennas
        theta_e_deg : Eve angle [degrees]
        sigma_alpha : path loss amplitude
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
    d_k_min:  float  = 1.0,
    d_k_max:  float  = 1.0,
    d_e:      float  = 1.0,
    alpha_pl: float  = 0.0,
    rng  = None,
    seed = None,
):
    """
    Generate one mmWave channel realization with optional path loss.

    CU channels : Rician (Su et al. Eq.2) + path loss sigma_cu
    Eve channel : Geometric g_e = alpha_k * b(theta_e) + path loss sigma_e

    Path loss model: sigma = (d_0/d)^(alpha_pl/2)

    Backward compatible: alpha_pl=0 gives no path loss (default).

    Parameters:
        N_t              : transmit antennas
        N                : number of CUs
        kappa            : Rician K-factor
        L_p              : NLoS paths per CU
        sigma_alpha      : Eve path-loss std dev (before distance scaling)
        theta_E_min_deg  : Eve angle min [degrees]
        theta_E_max_deg  : Eve angle max [degrees]
        cu_angle_min_deg : CU LoS angle min [degrees]
        cu_angle_max_deg : CU LoS angle max [degrees]
        d_0              : reference distance [m]
        d_k_min          : CU min distance [m]
        d_k_max          : CU max distance [m]
        d_e              : Eve distance [m]
        alpha_pl         : path loss exponent
        rng              : numpy random generator
        seed             : random seed

    Returns:
        dict: H (N_t,N), g_e (N_t,), theta_E (float)
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    theta_E = float(rng.uniform(theta_E_min_deg, theta_E_max_deg))

    H = np.zeros((N_t, N), dtype=complex)
    for i in range(N):
        angle_los = float(rng.uniform(cu_angle_min_deg, cu_angle_max_deg))
        d_k       = float(rng.uniform(d_k_min, d_k_max))
        sigma_cu  = (d_0/d_k)**(alpha_pl/2) if alpha_pl > 0 else 1.0
        H[:, i]   = generate_cu_channel(N_t, kappa, L_p, angle_los, sigma_cu, rng)

    sigma_e = (d_0/d_e)**(alpha_pl/2) if alpha_pl > 0 else 1.0
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

    d_0      = 20.0
    d_k_min  = 40.0
    d_k_max  = 60.0
    d_e      = 20.0
    alpha_pl = 2.5

    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from beamforming import zf_precoder, an_covariance_isotropic, null_space_projector
    from signal_model import compute_sinr_cu, compute_sinr_eve, compute_secrecy_rate
    from itertools import combinations

    print("=" * 55)
    print("  channel_mmwave.py — path loss verification")
    print("=" * 55)

    rng = np.random.default_rng(42)

    ch = generate_channels_mmwave(
        N_t=N_t, N=N, kappa=kappa, L_p=L_p,
        sigma_alpha=1.0,
        theta_E_min_deg=-30.0, theta_E_max_deg=30.0,
        d_0=d_0, d_k_min=d_k_min, d_k_max=d_k_max,
        d_e=d_e, alpha_pl=alpha_pl,
        seed=42,
    )

    norms_cu = np.array([np.sum(np.abs(ch["H"][:,k])**2) for k in range(N)])
    norm_eve = np.sum(np.abs(ch["g_e"])**2)

    print(f"\n[1] CU channel norms ||h_k||²:")
    print(f"    mean={norms_cu.mean():.4f}  "
          f"min={norms_cu.min():.4f}  "
          f"max={norms_cu.max():.4f}")
    print(f"    Expected ≈ sigma_cu² * N_t ≈ 0.5-2.0")

    print(f"\n[2] Eve channel norm ||g_e||²:")
    print(f"    {norm_eve:.4f}  "
          f"(d_e=d_0 → sigma_e=1.0 → E[||g_e||²] = N_t = {N_t})")

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
                d_0=d_0, d_k_min=d_k_min, d_k_max=d_k_max,
                d_e=d_e, alpha_pl=alpha_pl,
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

    print(f"\n    Target: 3-9 bits/s/Hz (Su et al. Fig.6/8)")

    print(f"\n[4] Backward compatibility (no path loss):")
    ch3 = generate_channels_mmwave(
        N_t=N_t, N=N, kappa=kappa, L_p=L_p,
        sigma_alpha=1.0,
        theta_E_min_deg=-30.0, theta_E_max_deg=30.0,
        seed=42,
    )
    norms3 = np.array([np.sum(np.abs(ch3["H"][:,k])**2) for k in range(N)])
    print(f"    CU norm mean (no path loss): {norms3.mean():.2f}")
    print(f"    ✅ backward compatible")