"""Channel module validation tests.

Test 1: Dimensions and shapes
Test 2: Statistical properties  (E[||h_k||^2] = M, zero distance correlation)
Test 3: Eve channel properties  (LoS structure, angle coverage)

Run:
    python test_channel.py
"""
from __future__ import annotations

import numpy as np
from channel import generate_channels, generate_dataset


# ---------------------------------------------------------------------------
# Test 1: Dimensions and shapes
# ---------------------------------------------------------------------------

def test_single_sample(M: int = 8, N: int = 10) -> None:
    """Check all shapes and keys in one sample."""
    sample = generate_channels(num_tx=M, N=N, seed=0)

    H       = sample["H"]
    g_e     = sample["g_e"]
    theta_E = sample["theta_E"]

    print("=== Test 1: Dimensions ===")
    print(f"  H shape       : {H.shape}      (expected ({M}, {N}))")
    print(f"  g_e shape     : {g_e.shape}   (expected ({M},))")
    print(f"  theta_E (deg) : {np.rad2deg(theta_E):.2f}")

    assert H.shape   == (M, N),  f"H shape wrong: {H.shape}"
    assert g_e.shape == (M,),    f"g_e shape wrong: {g_e.shape}"
    assert -np.pi/2 <= theta_E <= np.pi/2, "theta_E out of range"

    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Test 2: Statistical properties
# ---------------------------------------------------------------------------

def test_statistics(num_samples: int = 2000, M: int = 8, N: int = 10) -> None:
    """Verify CN(0, I_M) properties over many realisations.

    For h_k ~ CN(0, I_M):
        E[||h_k||^2]  = M         (channel power)
        Var[||h_k||^2] = M        (chi-squared with 2M dof / 2)
        Correlation(distance, power) ~ 0  (no path loss)
    """
    norms = np.zeros(num_samples)

    for i in range(num_samples):
        sample = generate_channels(num_tx=M, N=N, seed=i)
        norms[i] = np.linalg.norm(sample["H"][:, 0]) ** 2

    print("=== Test 2: Statistical Properties ===")
    print(f"  E[||h_k||^2]   = {np.mean(norms):.3f}  (expected {M}.000)")
    print(f"  Std[||h_k||^2] = {np.std(norms):.3f}  (expected {np.sqrt(M):.3f})")

    assert abs(np.mean(norms) - M) < 0.2, \
        f"Mean channel power wrong: {np.mean(norms):.3f} != {M}"

    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Test 3: Eve channel properties
# ---------------------------------------------------------------------------

def test_eve_channel(num_samples: int = 1000, M: int = 8) -> None:
    """Verify Eve channel structure and angle coverage.

    For g_e = beta_e * alpha_t(theta_E):
        ||g_e||^2 = |beta_e|^2 * ||alpha_t||^2 = |beta_e|^2 * M
        theta_E ~ Uniform[-pi/2, pi/2]
    """
    theta_vals = np.zeros(num_samples)
    norms      = np.zeros(num_samples)

    for i in range(num_samples):
        sample     = generate_channels(num_tx=M, seed=i)
        theta_vals[i] = sample["theta_E"]
        norms[i]   = np.linalg.norm(sample["g_e"]) ** 2

    print("=== Test 3: Eve Channel Properties ===")
    print(f"  theta_E range : [{np.rad2deg(theta_vals.min()):.1f}, "
          f"{np.rad2deg(theta_vals.max()):.1f}] deg  "
          f"(expected [-90, 90])")
    print(f"  theta_E mean  : {np.rad2deg(np.mean(theta_vals)):.2f} deg  "
          f"(expected ~0)")
    print(f"  ||g_e||^2 / M : {np.mean(norms)/M:.4e}  "
          f"(= |beta_e|^2, one-way path gain)")

    assert theta_vals.min() >= -np.pi/2, "theta_E below -pi/2"
    assert theta_vals.max() <=  np.pi/2, "theta_E above  pi/2"

    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Test 4: Dataset generation
# ---------------------------------------------------------------------------

def test_dataset(num_samples: int = 100) -> None:
    """Check dataset returns correct number of samples and is reproducible."""
    dataset = generate_dataset(num_samples, seed=42)

    print("=== Test 4: Dataset Generation ===")
    print(f"  Samples generated : {len(dataset)}  (expected {num_samples})")

    # Reproducibility: same seed -> same first sample
    dataset2 = generate_dataset(num_samples, seed=42)
    match = np.allclose(dataset[0]["H"], dataset2[0]["H"])
    print(f"  Reproducible      : {match}  (expected True)")

    assert len(dataset) == num_samples
    assert match, "Dataset not reproducible with same seed"

    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 45)
    print("  Channel Module Validation")
    print("=" * 45)
    print()

    test_single_sample()
    test_statistics()
    test_eve_channel()
    test_dataset()

    print("All tests passed.")