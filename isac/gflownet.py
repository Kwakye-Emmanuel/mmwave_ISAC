from __future__ import annotations

import math
from typing import Callable, List, NamedTuple, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from config import GFlowNetConfig


class _Transition(NamedTuple):
    s:      torch.Tensor   # (N,) binary state before action
    a:      int            # user index selected
    s_next: torch.Tensor   # (N,) binary state after action


class _PolicyNet(nn.Module):
    """Shared MLP producing forward and backward logits.

    Input  : [s || snr]  in R^{2N}
    Output : [P_F logits || P_B logits]  each in R^N
    """

    def __init__(self, N: int, hidden: int):
        super().__init__()
        self.N = N
        self.net = nn.Sequential(
            nn.Linear(2 * N, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * N),
        )

    def forward(
        self, s: torch.Tensor, snr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(torch.cat([s, snr], dim=-1))
        return out[..., :self.N], out[..., self.N:]


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = logits.clone()
    logits[~mask] = -1e9
    return F.log_softmax(logits, dim=-1)


class GFlowNet:
    """GFlowNet for K-user scheduling trained with Trajectory Balance loss.

    The MDP:
        state   s in {0,1}^N  — binary user selection mask
        action  a in {0..N-1} — add one unselected user
        depth   K steps       — terminal when |s| = K
        reward  R(s) >= 0     — provided externally (e.g. sum-rate)

    At convergence:
        P(terminal = K_set)  proportional to  R(K_set)

    TB loss (Malkin et al., NeurIPS 2022):
        L(tau) = [lnZ + sum_t ln P_F(a_t|s_t) - ln R(x) - sum_t ln P_B(a_t|s_{t+1})]^2
    """

    def __init__(self, cfg: GFlowNetConfig):
        self.cfg = cfg
        self.policy = _PolicyNet(cfg.N, cfg.hidden)
        self.log_z = nn.Parameter(torch.zeros(1))
        self.opt = Adam(list(self.policy.parameters()) + [self.log_z], lr=cfg.lr)

    def _temperature(self, ep: int) -> float:
        frac = min(ep / (self.cfg.n_episodes * 0.8), 1.0)
        return self.cfg.temp_start + frac * (self.cfg.temp_end - self.cfg.temp_start)

    @torch.no_grad()
    def _rollout(
        self, snr: torch.Tensor, temp: float
    ) -> Tuple[List[_Transition], Tuple[int, ...]]:
        s = torch.zeros(self.cfg.N)
        transitions: List[_Transition] = []

        for _ in range(self.cfg.K):
            fwd, _ = self.policy(s.unsqueeze(0), snr.unsqueeze(0))
            fwd = fwd.squeeze(0) / max(temp, 1e-6)

            log_pf = _masked_log_softmax(fwd, s < 0.5)
            a = int(torch.multinomial(log_pf.exp(), 1).item())

            s_next = s.clone()
            s_next[a] = 1.0
            transitions.append(_Transition(s.clone(), a, s_next.clone()))
            s = s_next

        selected = tuple(sorted(t.a for t in transitions))
        return transitions, selected

    def _tb_loss(
        self,
        transitions: List[_Transition],
        snr: torch.Tensor,
        reward: float,
        temp: float,
    ) -> torch.Tensor:
        """Recompute log P_F and log P_B with gradients, then compute TB loss.

        All K states are batched into two forward passes instead of 2K passes.
        """
        K = len(transitions)
        snr_exp = snr.unsqueeze(0).expand(K, -1)

        s_batch      = torch.stack([t.s      for t in transitions])
        s_next_batch = torch.stack([t.s_next for t in transitions])

        fwd_batch, _  = self.policy(s_batch,      snr_exp)
        _,  bwd_batch = self.policy(s_next_batch, snr_exp)
        fwd_batch = fwd_batch / max(temp, 1e-6)

        log_pf_sum = torch.zeros(1)
        log_pb_sum = torch.zeros(1)

        for i, t in enumerate(transitions):
            log_pf_sum = log_pf_sum + _masked_log_softmax(fwd_batch[i], t.s < 0.5)[t.a]
            log_pb_sum = log_pb_sum + _masked_log_softmax(bwd_batch[i], t.s_next > 0.5)[t.a]

        log_r = math.log(max(reward, self.cfg.reward_floor))
        return (self.log_z + log_pf_sum - log_r - log_pb_sum).pow(2)

    def train(
        self,
        reward_fn: Callable[[Tuple[int, ...], np.ndarray], float],
        snr_sampler: Callable[[], np.ndarray],
        log_every: int = 1000,
    ) -> Tuple[List[float], List[float]]:
        """Train for cfg.n_episodes episodes.

        Args:
            reward_fn   : R(selected_set, snr_array) -> float >= 0
            snr_sampler : () -> SNR array of shape (N,)
            log_every   : print interval
        Returns:
            losses      : per-episode TB loss values
            lnz_history : lnZ recorded at each log_every checkpoint
        """
        losses:      List[float] = []
        lnz_history: List[float] = []

        for ep in range(self.cfg.n_episodes):
            snr_np = snr_sampler()
            snr    = torch.tensor(snr_np, dtype=torch.float32)
            temp   = self._temperature(ep)

            transitions, selected = self._rollout(snr, temp)
            reward = reward_fn(selected, snr_np)

            self.opt.zero_grad()
            loss = self._tb_loss(transitions, snr, reward, temp)
            loss.backward()
            self.opt.step()

            losses.append(loss.item())

            if (ep + 1) % log_every == 0:
                avg = np.mean(losses[-log_every:])
                lnz_history.append(self.log_z.item())
                print(
                    f"ep {ep+1:>7d}  "
                    f"loss={avg:.4f}  "
                    f"lnZ={self.log_z.item():.3f}  "
                    f"temp={temp:.3f}"
                )

        return losses, lnz_history

    @torch.no_grad()
    def sample(self, snr: np.ndarray, greedy: bool = False) -> Tuple[int, ...]:
        """Sample one user set from the trained policy.

        greedy=True uses near-zero temperature (argmax policy).
        """
        temp = 1e-4 if greedy else 1.0
        snr_t = torch.tensor(snr, dtype=torch.float32)
        _, selected = self._rollout(snr_t, temp)
        return selected

    @torch.no_grad()
    def evaluate(
        self,
        oracle_fn: Callable[[np.ndarray], Tuple[int, ...]],
        snr_sampler: Callable[[], np.ndarray],
        n_trials: int = 1000,
    ) -> float:
        """Fraction of trials where greedy sample matches oracle."""
        correct = sum(
            int(self.sample(snr := snr_sampler(), greedy=True) == oracle_fn(snr))
            for _ in range(n_trials)
        )
        return correct / n_trials


if __name__ == "__main__":
    import numpy as np
    from config import SystemConfig, GFlowNetConfig
    from channel_mmwave import generate_channels_mmwave
    from signal_model import compute_sinr_cu, compute_sinr_eve, compute_secrecy_rate

    # ----------------------------------------------------------------
    # Configs
    # ----------------------------------------------------------------
    sys_cfg = SystemConfig()
    gfn_cfg = GFlowNetConfig(N=sys_cfg.N, K=sys_cfg.K)
    gfn     = GFlowNet(gfn_cfg)

    rng = np.random.default_rng(sys_cfg.seed)

    # ----------------------------------------------------------------
    # Channel sampler — generates one full realization
    # returns channel norms ||h_k||^2 as input features (N,)
    # ----------------------------------------------------------------
    def channel_sampler():
        ch = generate_channels_mmwave(
            N_t             = sys_cfg.N_t,
            N               = sys_cfg.N,
            kappa           = sys_cfg.kappa,
            L_p             = sys_cfg.L_p,
            sigma_alpha     = sys_cfg.sigma_alpha,
            theta_E_min_deg = sys_cfg.theta_E_min_deg,
            theta_E_max_deg = sys_cfg.theta_E_max_deg,
            seed            = int(rng.integers(0, 2**31)),
        )
        # channel norms as input features
        norms = np.array([
            np.sum(np.abs(ch["H"][:, i])**2)
            for i in range(sys_cfg.N)
        ])
        return norms, ch

    # ----------------------------------------------------------------
    # Reward function — secrecy rate for selected user subset
    # ----------------------------------------------------------------
    from beamforming import zf_precoder, an_covariance_isotropic
    
    def reward_fn(selected, ch):
        H   = ch["H"]        # (N_t, N)
        g_e = ch["g_e"]      # (N_t,)
        K   = len(selected)

        # Extract scheduled user chanenels
        H_K = H[:, list(selected)]  # (N_t, K)

        # ZF precoder
        W = zf_precoder(H_K, sys_cfg.P_t, sys_cfg.rho, K)


        # Isotropic null-space AN
        R_N = an_covariance_isotropic(H_K, sys_cfg.N_t, sys_cfg.P_t, sys_cfg.rho, K)

        # Compute SINRs
        sinr_cu = [
            compute_sinr_cu(H[:, selected[i]], W, R_N, i, sys_cfg.sigma2_C)
            for i in range(K)
        ]
        sinr_eve = [
            compute_sinr_eve(g_e, W, R_N, i, sys_cfg.sigma2_e)
            for i in range(K)
        ]

        return compute_secrecy_rate(sinr_cu, sinr_eve)

    # ----------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------
    print(f"Training GFlowNet: N={sys_cfg.N}, K={sys_cfg.K}, "
          f"{gfn_cfg.n_episodes} episodes")

    # Wrap samplers to match GFlowNet interface
    ch_store = {}

    def snr_sampler():
        norms, ch = channel_sampler()
        ch_store["current"] = ch
        return norms

    def reward_wrapper(selected, norms):
        return reward_fn(selected, ch_store["current"])

    losses, lnz_history = gfn.train(
        reward_fn   = reward_wrapper,
        snr_sampler = snr_sampler,
        log_every   = gfn_cfg.log_every,
    )

    print(f"\nFinal lnZ: {gfn.log_z.item():.4f}")
    print(f"Final avg loss: {np.mean(losses[-1000:]):.4f}")


    # ----------------------------------------------------------------
    # Reward diagnostic — check reward distribution
    # ----------------------------------------------------------------
    print("\n--- Reward Diagnostic ---")
    rewards = []
    zero_count = 0

    for _ in range(1000):
        norms, ch = channel_sampler()
        # test all possible K=2 subsets
        from itertools import combinations
        subset_rewards = []
        for selected in combinations(range(sys_cfg.N), sys_cfg.K):
            r = reward_fn(selected, ch)
            subset_rewards.append(r)
            if r == 0.0:
                zero_count += 1
        rewards.extend(subset_rewards)

    rewards = np.array(rewards)
    total   = len(rewards)

    print(f"Total subsets evaluated : {total}")
    print(f"Zero rewards            : {zero_count} ({100*zero_count/total:.1f}%)")
    print(f"Non-zero rewards        : {total - zero_count} ({100*(total-zero_count)/total:.1f}%)")
    print(f"Mean reward             : {np.mean(rewards):.4f}")
    print(f"Max reward              : {np.max(rewards):.4f}")
    print(f"Min non-zero reward     : {rewards[rewards > 0].min():.4f} "
          f"(if any)")