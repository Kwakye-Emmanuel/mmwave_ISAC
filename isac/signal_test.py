"""Short validation tests for signal.py.

Run:
    python test_signal.py
"""
from __future__ import annotations

import numpy as np
from .channel import generate_channels
from .sensing import ula_steering
from .signal import (
    zf_precoder,
    an_covariance_uniform,
    an_covariance_directed,
    compute_sinr_user,
    compute_sinr_eve,
    compute_secrecy_rate,
    compute_secrecy_sum_rate,
)

M, N, Kd     = 8, 10, 2
P_t          = 1.0
sigma2_C     = 0.1
alpha        = 0.5
P_sig        = alpha * P_t
P_AN         = (1 - alpha) * P_t
sched        = [0, 1]

sample       = generate_channels(M=M, N=N, seed=0)
H            = sample["H"]
g_e          = sample["g_e"]
theta_E      = sample["theta_E"]
at_hat       = ula_steering(theta_E, M)
H_D          = H[:, sched].conj().T     # (Kd, M)
W_D          = zf_precoder(H_D, P_sig, Kd)
R_N_uni      = an_covariance_uniform(H_D, M, Kd, P_AN)
R_N_dir      = an_covariance_directed(H_D, M, P_AN, at_hat)


def test_zf_precoder():
    print("--- Test 1: ZF Precoder ---")

    # Inter-user interference must be zero
    interf = abs(H[:, 0].conj() @ W_D[:, 1]) ** 2
    print(f"  Inter-user interference : {interf:.2e}   (want ~0)")
    assert interf < 1e-20, "ZF interference too high"

    # Power per stream = P_sig / Kd
    for k in range(Kd):
        p = np.linalg.norm(W_D[:, k]) ** 2
        print(f"  ||w_{k}||^2 = {p:.4f}   (want {P_sig/Kd:.4f})")
        assert abs(p - P_sig / Kd) < 1e-10

    print("  PASSED\n")


def test_an_leakage():
    print("--- Test 2: AN Leakage to Users ---")

    for label, R_N in [("Uniform", R_N_uni), ("Directed", R_N_dir)]:
        for k in range(Kd):
            leak = abs(float(np.real(H[:, k].conj() @ R_N @ H[:, k])))
            print(f"  {label} AN leakage user {k}: {leak:.2e}   (want ~0)")
            assert leak < 1e-10, f"{label} AN leaks to user {k}"

    print("  PASSED\n")


def test_directed_vs_uniform_at_eve():
    print("--- Test 3: Directed AN > Uniform AN at Eve ---")

    # Use unit steering vector for clean comparison
    g_dir    = ula_steering(theta_E, M)
    an_uni   = float(np.real(g_dir.conj() @ R_N_uni @ g_dir))
    an_dir   = float(np.real(g_dir.conj() @ R_N_dir @ g_dir))
    gain     = an_dir / (an_uni + 1e-30)

    print(f"  Uniform  AN at Eve : {an_uni:.4f}")
    print(f"  Directed AN at Eve : {an_dir:.4f}")
    print(f"  Gain               : {gain:.2f}x   (expect <= {M - Kd}x = {M-Kd}x)")
    assert an_dir >= an_uni, "Directed AN should be >= uniform at Eve"

    print("  PASSED\n")


def test_secrecy_rates():
    print("--- Test 4: Secrecy Rates ---")

    R_B1 = compute_secrecy_sum_rate(
        H, g_e, sched, None,   P_t, sigma2_C, time_frac=1.0
    )
    R_B2 = compute_secrecy_sum_rate(
        H, g_e, sched, at_hat, P_t, sigma2_C, time_frac=0.68
    )

    print(f"  B1 (no sensing, full frame) : {R_B1:.4f} bps/Hz")
    print(f"  B2 (sensing, time_frac=0.68): {R_B2:.4f} bps/Hz")
    assert R_B1 >= 0, "B1 secrecy rate negative"
    assert R_B2 >= 0, "B2 secrecy rate negative"

    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 40)
    print("  Signal Module Tests")
    print("=" * 40)
    print()

    test_zf_precoder()
    test_an_leakage()
    test_directed_vs_uniform_at_eve()
    test_secrecy_rates()

    print("All tests passed.")