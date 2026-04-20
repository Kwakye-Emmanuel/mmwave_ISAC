"""Scheduling algorithms for ISAC-aided physical layer security.

Four schedulers:

    random_scheduling        : random subset selection (B1 and B2 baseline)
    oracle_scheduling_genie  : brute-force optimal using TRUE g_e (upper bound)
                               → used in simulate.py for Fig. 2 Genie-Aided curve
    oracle_scheduling_label  : brute-force optimal using ESTIMATED g_hat_e for AN
                               → used in dataset.py for DL training labels
    mask_to_indices          : convert binary mask to index list
    indices_to_mask          : convert index list to binary mask

Key distinction:
    oracle_scheduling_genie  : AN design uses g_e  (perfect — unachievable)
    oracle_scheduling_label  : AN design uses g_hat_e (estimated — matches inference)

    Both evaluate secrecy with TRUE g_e for honest performance assessment.
    Training labels must use g_hat_e for AN to be consistent with DL inference.

Reference:
    SecureLEO lab code (scheduling.py) — adapted for terrestrial ISAC.
"""
from __future__ import annotations

import itertools

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from .signal import compute_secrecy_sum_rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def indices_to_mask(indices: list[int], N: int) -> NDArray[np.floating]:
    """Convert list of selected indices to binary mask.

    Args:
        indices : selected user indices (0-based)
        N       : total number of users
    Returns:
        mask : (N,) binary float32 array
    """
    mask = np.zeros(N, dtype=np.float32)
    mask[indices] = 1.0
    return mask


def mask_to_indices(mask: NDArray[np.floating]) -> list[int]:
    """Convert binary mask to list of selected indices.

    Args:
        mask : (N,) binary array
    Returns:
        indices : list of selected user indices
    """
    return list(np.where(mask > 0.5)[0])


# ---------------------------------------------------------------------------
# Random scheduling  (B1 and B2 baseline)
# ---------------------------------------------------------------------------

def random_scheduling(
    N:   int,
    Kd:  int,
    rng: Generator | None = None,
) -> NDArray[np.floating]:
    """Random user selection baseline.

    Selects Kd users uniformly at random from N.
    Used by B1 (no sensing) and B2 (sensing-assisted BF, random sched.).

    Args:
        N   : total number of users
        Kd  : number of users to schedule
        rng : random generator
    Returns:
        mask : (N,) binary selection mask
    """
    if rng is None:
        rng = np.random.default_rng()
    selected = rng.choice(N, size=Kd, replace=False)
    return indices_to_mask(list(selected), N)


# ---------------------------------------------------------------------------
# Genie-Aided Oracle  (Fig. 2 upper bound — simulate.py only)
# ---------------------------------------------------------------------------

def oracle_scheduling_genie(
    H:        NDArray[np.complexfloating],
    g_e:      NDArray[np.complexfloating],
    Kd:       int,
    P_t:      float,
    sigma2_C: float,
    time_frac: float,
    alpha:    float = 0.5,
) -> tuple[NDArray[np.floating], float]:
    """Genie-aided oracle: theoretical upper bound for Fig. 2.

    Uses TRUE g_e for BOTH AN design and secrecy evaluation.
    The genie gives the BS perfect Eve CSI — unachievable in practice.

    This represents the absolute performance ceiling:
        - AN is perfectly aimed at Eve using g_e
        - Scheduling picks the best D knowing exact Eve location
        - Secrecy is evaluated with true g_e

    DO NOT use for training label generation — see oracle_scheduling_label.

    Args:
        H         : (M, N) user channel matrix
        g_e       : (M,)   TRUE Eve channel  [genie provides this]
        Kd        : number of users to schedule
        P_t       : total transmit power [W]
        sigma2_C  : communication noise variance [W]
        time_frac : frame time fraction T_c/T
        alpha     : power split ratio rho
    Returns:
        best_mask : (N,) binary selection mask
        best_rate : best secrecy sum-rate [bps/Hz]
    """
    N         = H.shape[1]
    best_rate = -1.0
    best_mask = np.zeros(N, dtype=np.float32)

    for combo in itertools.combinations(range(N), Kd):
        sched_idx = list(combo)
        rate = compute_secrecy_sum_rate(
            H         = H,
            g_e       = g_e,
            sched_idx = sched_idx,
            g_hat_e   = g_e,        # ← genie: perfect AN direction
            P_t       = P_t,
            sigma2_C  = sigma2_C,
            time_frac = time_frac,
            rho       = alpha,
        )
        if rate > best_rate:
            best_rate = rate
            best_mask = indices_to_mask(sched_idx, N)

    return best_mask, float(best_rate)


# ---------------------------------------------------------------------------
# Label Oracle  (dataset.py training labels only)
# ---------------------------------------------------------------------------

def oracle_scheduling_label(
    H:        NDArray[np.complexfloating],
    g_e:      NDArray[np.complexfloating],
    Kd:       int,
    g_hat_e:  NDArray[np.complexfloating],
    P_t:      float,
    sigma2_C: float,
    time_frac: float,
    alpha:    float = 0.5,
) -> tuple[NDArray[np.floating], float]:
    """Label oracle for DL training — consistent with inference conditions.

    Uses ESTIMATED g_hat_e for AN design — exactly as the DL model will
    do at inference time. Uses TRUE g_e only for honest evaluation.

    Train/test consistency:
        Label oracle  → g_hat_e for AN  (matches DL inference) ✅
        DL inference  → g_hat_e for AN                         ✅

    If g_e were used for AN here, the model would learn scheduling
    decisions that are optimal under perfect CSI but suboptimal under
    estimated CSI — a train/test mismatch.

    DO NOT use for Fig. 2 Genie-Aided curve — see oracle_scheduling_genie.

    Args:
        H         : (M, N) user channel matrix
        g_e       : (M,)   TRUE Eve channel  [evaluation only]
        Kd        : number of users to schedule
        g_hat_e   : (M,)   ESTIMATED Eve channel from sensing stage
        P_t       : total transmit power [W]
        sigma2_C  : communication noise variance [W]
        time_frac : frame time fraction T_c/T
        alpha     : power split ratio rho
    Returns:
        best_mask : (N,) binary selection mask
        best_rate : best secrecy sum-rate [bps/Hz]
    """
    N         = H.shape[1]
    best_rate = -1.0
    best_mask = np.zeros(N, dtype=np.float32)

    for combo in itertools.combinations(range(N), Kd):
        sched_idx = list(combo)
        rate = compute_secrecy_sum_rate(
            H         = H,
            g_e       = g_e,
            sched_idx = sched_idx,
            g_hat_e   = g_hat_e,    # ← estimated: consistent with DL inference
            P_t       = P_t,
            sigma2_C  = sigma2_C,
            time_frac = time_frac,
            rho       = alpha,
        )
        if rate > best_rate:
            best_rate = rate
            best_mask = indices_to_mask(sched_idx, N)

    return best_mask, float(best_rate)