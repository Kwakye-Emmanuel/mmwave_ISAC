"""GFlowNet for sensing-aided secure user scheduling.

Implements Trajectory Balance (TB) GFlowNet for K-user scheduling
in an ISAC physical layer security system.

Phase 1: Discrete GFlowNet + Deep Sets policy + fixed ZF beamforming
         Sensing state (theta_hat, crb) feeds into scheduling decision
         Reward = MC ergodic secrecy sum-rate over CRB uncertainty

Architecture: Deep Sets (Zaheer et al., NeurIPS 2017)
    - Permutation invariant by design
    - Per-user encoder with shared weights
    - Global context via mean aggregation
    - Per-user decoder with global context

Reference:
    Malkin et al., "Trajectory Balance: Improved Credit Assignment
    in GFlowNets", NeurIPS 2022.

    Su et al., "Sensing-Assisted Eavesdropper Estimation",
    IEEE TWC 2024.

    Zaheer et al., "Deep Sets", NeurIPS 2017.
"""
from __future__ import annotations

import math
from typing import List, NamedTuple, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from config import SystemConfig, GFlowNetConfig
from channel_mmwave import generate_channels_mmwave, generate_eve_channel
from beamforming import zf_precoder, an_covariance_isotropic, an_covariance_directed
from signal_model import compute_sinr_cu, compute_sinr_eve, compute_secrecy_rate
from sensing import run_sensing, steering_vector


# ---------------------------------------------------------------
# MDP transition
# ---------------------------------------------------------------

class _Transition(NamedTuple):
    s:      torch.Tensor   # (N,) binary state before action
    a:      int            # user index selected
    s_next: torch.Tensor   # (N,) binary state after action


# ---------------------------------------------------------------
# Feature builder — per-user features + Eve context
# ---------------------------------------------------------------

def build_features(
    H             : np.ndarray,   # (N_t, N)
    sensing_state : dict,
    N             : int,
) -> np.ndarray:
    """Build per-user feature matrix and Eve context vector.

    Per-user features (3 per user):
        f1_k = ||h_k||² / max_j||h_j||²       relative channel strength
        f2_k = |b(θ̂_e)^H h_k|² / ||h_k||²   Eve alignment (low = good)
        f3_k = f1_k / (f2_k + 1e-6)            secrecy potential

    Eve context (2):
        theta_hat_norm = theta_hat / 90.0       [-1, 1]
        crb_norm       = log10(crb + 1e-20)     log scale

    Returns:
        user_feats : (N, 3) per-user feature matrix
        eve_ctx    : (2,)   Eve context vector
    """
    N_t       = H.shape[0]
    theta_hat = sensing_state["theta_hat"]
    crb       = sensing_state["crb"]

    # Eve steering vector at estimated angle
    b_eve = steering_vector(theta_hat, N_t)    # (N_t,)

    # per-user features
    norms = np.array([np.sum(np.abs(H[:, k])**2) for k in range(N)])
    max_norm = norms.max() + 1e-10

    user_feats = np.zeros((N, 3), dtype=np.float32)
    for k in range(N):
        h_k  = H[:, k]
        n_k  = norms[k]

        # f1: relative channel strength
        f1 = n_k / max_norm

        # f2: Eve alignment — how much Eve sees user k's signal
        f2 = float(abs(b_eve.conj() @ h_k)**2) / (n_k + 1e-10)

        # f3: secrecy potential — strong channel, low Eve alignment
        f3 = f1 / (f2 + 1e-6)

        user_feats[k] = [f1, f2, f3]

    # Eve context
    eve_ctx = np.array([
        theta_hat / 90.0,
        np.log10(crb + 1e-20),
    ], dtype=np.float32)

    return user_feats, eve_ctx


# ---------------------------------------------------------------
# Deep Sets policy network
# ---------------------------------------------------------------

class _PolicyNet(nn.Module):
    """Deep Sets policy network — permutation invariant.

    Architecture:
        1. Per-user encoder (shared MLP):
           input: [user_feats_k(3) || eve_ctx(2) || s_k(1)] = (6,)
           → user_embedding_k  (hidden,)

        2. Aggregator:
           context = mean(user_embedding_k)  (hidden,)
           permutation invariant

        3. Per-user decoder:
           input: [user_embedding_k(hidden) || context(hidden)] = (2*hidden,)
           → [fwd_logit_k, bwd_logit_k]  (2,)

    Total parameters: much fewer than flat MLP but more expressive
    for set-structured inputs.
    """

    def __init__(self, N: int, hidden: int):
        super().__init__()
        self.N = N

        # shared per-user encoder
        # input: user_feat(3) + eve_ctx(2) + selection_state(1) = 6
        self.encoder = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

        # per-user decoder → forward and backward logits
        # input: user_embedding(hidden) + context(hidden) = 2*hidden
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),   # [fwd_logit, bwd_logit]
        )

    def forward(
        self,
        s:          torch.Tensor,   # (B, N) or (N,) binary state
        user_feats: torch.Tensor,   # (B, N, 3) or (N, 3)
        eve_ctx:    torch.Tensor,   # (B, 2) or (2,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            fwd_logits : (..., N) forward policy logits
            bwd_logits : (..., N) backward policy logits
        """
        # handle both batched and unbatched inputs
        batched = s.dim() == 2
        if not batched:
            s          = s.unsqueeze(0)           # (1, N)
            user_feats = user_feats.unsqueeze(0)  # (1, N, 3)
            eve_ctx    = eve_ctx.unsqueeze(0)      # (1, 2)

        B = s.shape[0]

        # expand eve_ctx and s to per-user
        # eve_ctx: (B, 2) → (B, N, 2)
        eve_exp = eve_ctx.unsqueeze(1).expand(B, self.N, 2)
        # s: (B, N) → (B, N, 1)
        s_exp   = s.unsqueeze(-1)

        # per-user encoder input: (B, N, 6)
        enc_input = torch.cat([user_feats, eve_exp, s_exp], dim=-1)

        # per-user embeddings: (B, N, hidden)
        embeddings = self.encoder(enc_input)

        # global context: (B, hidden) — mean aggregation
        context = embeddings.mean(dim=1)           # permutation invariant

        # expand context: (B, hidden) → (B, N, hidden)
        ctx_exp = context.unsqueeze(1).expand(B, self.N, embeddings.shape[-1])

        # per-user decoder input: (B, N, 2*hidden)
        dec_input = torch.cat([embeddings, ctx_exp], dim=-1)

        # per-user logits: (B, N, 2)
        logits = self.decoder(dec_input)

        fwd_logits = logits[..., 0]   # (B, N)
        bwd_logits = logits[..., 1]   # (B, N)

        if not batched:
            fwd_logits = fwd_logits.squeeze(0)   # (N,)
            bwd_logits = bwd_logits.squeeze(0)   # (N,)

        return fwd_logits, bwd_logits


def _masked_log_softmax(
    logits: torch.Tensor,
    mask:   torch.Tensor,
) -> torch.Tensor:
    logits = logits.clone()
    logits[~mask] = -1e9
    return F.log_softmax(logits, dim=-1)


# ---------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------

def compute_reward(
    selected  : Tuple[int, ...],
    H         : np.ndarray,
    theta_hat : float,
    crb       : float,
    P_t       : float,
    sys_cfg   : SystemConfig,
    rng       : np.random.Generator,
    n_mc      : int = 100,
    an_type   : str = "directed",        # "directed" or "isotropic"
    g_e_hat   : np.ndarray | None = None,
) -> float:
    """Ergodic secrecy sum-rate via Monte Carlo over CRB uncertainty.

    R(K) = (1/n_mc) sum_m [sum_k [log2(1+SINR_k^CU) - log2(1+SINR_k^E)]^+]

    an_type = "directed"  → rank-1 AN toward g_e_hat (proposed scheme)
    an_type = "isotropic" → uniform null-space AN (DLS baseline)

    sinr_cu computed once — fixed for given H, W, R_N
    sinr_eve computed per MC sample — depends on g_e_sample
    """
    K   = len(selected)
    H_K = H[:, list(selected)]

    # beamforming
    W = zf_precoder(H_K, P_t, sys_cfg.phi, K)

    if an_type == "directed" and g_e_hat is not None:
        R_N = an_covariance_directed(H_K, g_e_hat, sys_cfg.N_t, P_t,
                                      sys_cfg.phi, K)
    else:
        R_N = an_covariance_isotropic(H_K, sys_cfg.N_t, P_t, sys_cfg.phi, K)

    # sinr_cu fixed — does not depend on Eve channel
    sinr_cu = [
        compute_sinr_cu(H[:, selected[i]], W, R_N, i, sys_cfg.sigma2_C)
        for i in range(K)
    ]

    # CRB std in degrees
    crb_std_deg = float(np.sqrt(crb) * 180.0 / np.pi)

    rates = []
    for _ in range(n_mc):
        theta_sample = float(np.clip(
            rng.normal(theta_hat, crb_std_deg), -90.0, 90.0
        ))
        g_e_sample = generate_eve_channel(
            sys_cfg.N_t, theta_sample, sys_cfg.sigma_alpha, rng
        )
        sinr_eve = [
            compute_sinr_eve(g_e_sample, W, R_N, i, sys_cfg.sigma2_e)
            for i in range(K)
        ]
        rates.append(compute_secrecy_rate(sinr_cu, sinr_eve))

    return float(np.mean(rates))


# ---------------------------------------------------------------
# GFlowNet
# ---------------------------------------------------------------

class GFlowNet:
    """GFlowNet for K-user scheduling with TB loss and Deep Sets policy.

    MDP:
        state   s in {0,1}^N
        action  a in {0..N-1}
        depth   K steps
        reward  R(K) = MC ergodic secrecy sum-rate

    Policy: Deep Sets — permutation invariant
    """

    def __init__(self, cfg: GFlowNetConfig):
        self.cfg    = cfg
        self.policy = _PolicyNet(cfg.N, cfg.hidden)
        self.log_z  = nn.Parameter(torch.zeros(1))
        self.opt    = Adam(
            list(self.policy.parameters()) + [self.log_z],
            lr=cfg.lr,
        )

    def _temperature(self, ep: int) -> float:
        frac = min(ep / (self.cfg.n_episodes * 0.8), 1.0)
        return self.cfg.temp_start + frac * (
            self.cfg.temp_end - self.cfg.temp_start
        )

    @torch.no_grad()
    def _rollout(
        self,
        user_feats: torch.Tensor,   # (N, 3)
        eve_ctx:    torch.Tensor,   # (2,)
        temp:       float,
    ) -> Tuple[List[_Transition], Tuple[int, ...]]:
        s           = torch.zeros(self.cfg.N)
        transitions = []

        for _ in range(self.cfg.K):
            fwd, _ = self.policy(s, user_feats, eve_ctx)
            fwd    = fwd / max(temp, 1e-6)

            log_pf = _masked_log_softmax(fwd, s < 0.5)
            a      = int(torch.multinomial(log_pf.exp(), 1).item())

            s_next    = s.clone()
            s_next[a] = 1.0
            transitions.append(_Transition(s.clone(), a, s_next.clone()))
            s = s_next

        selected = tuple(sorted(t.a for t in transitions))
        return transitions, selected

    def _tb_loss(
        self,
        transitions: List[_Transition],
        user_feats:  torch.Tensor,
        eve_ctx:     torch.Tensor,
        reward:      float,
        temp:        float,
    ) -> torch.Tensor:
        K = len(transitions)

        # batch all states for efficiency
        s_batch  = torch.stack([t.s      for t in transitions])  # (K, N)
        sn_batch = torch.stack([t.s_next for t in transitions])  # (K, N)

        # expand user_feats and eve_ctx for batch
        uf_exp  = user_feats.unsqueeze(0).expand(K, -1, -1)  # (K, N, 3)
        ec_exp  = eve_ctx.unsqueeze(0).expand(K, -1)          # (K, 2)

        fwd_batch, _  = self.policy(s_batch,  uf_exp, ec_exp)
        _,  bwd_batch = self.policy(sn_batch, uf_exp, ec_exp)
        fwd_batch     = fwd_batch / max(temp, 1e-6)

        log_pf_sum = torch.zeros(1)
        log_pb_sum = torch.zeros(1)

        for i, t in enumerate(transitions):
            log_pf_sum = log_pf_sum + \
                _masked_log_softmax(fwd_batch[i], t.s      < 0.5)[t.a]
            log_pb_sum = log_pb_sum + \
                _masked_log_softmax(bwd_batch[i], t.s_next > 0.5)[t.a]

        log_r = math.log(max(reward, self.cfg.reward_floor))
        return (self.log_z + log_pf_sum - log_r - log_pb_sum).pow(2)

    def train(
        self,
        sys_cfg:   SystemConfig,
        P_t:       float,
        rng:       np.random.Generator,
        log_every: int = 1000,
    ) -> Tuple[List[float], List[float]]:
        losses:      List[float] = []
        lnz_history: List[float] = []

        for ep in range(self.cfg.n_episodes):
            temp = self._temperature(ep)

            # 1. channel realization
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
            H       = ch["H"]
            theta_E = ch["theta_E"]

            # 2. sensing
            beta = sys_cfg.beta_mag * np.exp(1j * rng.uniform(0, 2*np.pi))
            sensing_state = run_sensing(
                theta_E_deg = theta_E,
                beta        = beta,
                N_t         = sys_cfg.N_t,
                N_r         = sys_cfg.N_r,
                L           = sys_cfg.L,
                P0          = P_t,
                sigma2_R    = sys_cfg.sigma2_R,
                seed        = int(rng.integers(0, 2**31)),
            )

            # 3. build features
            user_feats_np, eve_ctx_np = build_features(
                H, sensing_state, sys_cfg.N
            )
            user_feats = torch.tensor(user_feats_np, dtype=torch.float32)
            eve_ctx    = torch.tensor(eve_ctx_np,    dtype=torch.float32)

            # 4. rollout
            transitions, selected = self._rollout(user_feats, eve_ctx, temp)

            # 5. MC reward
            reward = compute_reward(
                selected  = selected,
                H         = H,
                theta_hat = sensing_state["theta_hat"],
                crb       = sensing_state["crb"],
                P_t       = P_t,
                sys_cfg   = sys_cfg,
                rng       = rng,
                n_mc      = self.cfg.n_mc,
                an_type   = "directed",
                g_e_hat   = sensing_state["g_e_hat"],
            )

            # 6. TB loss
            self.opt.zero_grad()
            loss = self._tb_loss(
                transitions, user_feats, eve_ctx, reward, temp
            )
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
                    f"temp={temp:.3f}  "
                    f"reward={reward:.4f}"
                )

        return losses, lnz_history

    @torch.no_grad()
    def sample(
        self,
        user_feats: np.ndarray,   # (N, 3)
        eve_ctx:    np.ndarray,   # (2,)
        greedy:     bool = False,
    ) -> Tuple[int, ...]:
        temp      = 1e-4 if greedy else 1.0
        uf_t = torch.tensor(user_feats, dtype=torch.float32)
        ec_t = torch.tensor(eve_ctx,    dtype=torch.float32)
        _, selected = self._rollout(uf_t, ec_t, temp)
        return selected


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

if __name__ == "__main__":

    sys_cfg = SystemConfig()
    gfn_cfg = GFlowNetConfig(N=sys_cfg.N, K=sys_cfg.K)
    gfn     = GFlowNet(gfn_cfg)
    rng     = np.random.default_rng(sys_cfg.seed)

    #P0 - Power budget
    P_t_train = 10**(30.0/10.0) * 1e-3

    print(f"{'='*60}")
    print(f"  GFlowNet Training — Phase 1 (Deep Sets)")
    print(f"{'='*60}")
    print(f"  N={sys_cfg.N}, K={sys_cfg.K}, N_t={sys_cfg.N_t}")
    print(f"  P0_train=30.0 dBm, P0={P_t_train:.4e} W")
    print(f"  Architecture: Deep Sets (permutation invariant)")
    print(f"  Features: relative norm, Eve alignment, secrecy potential")
    print(f"{'='*60}\n")

    losses, lnz_history = gfn.train(
        sys_cfg   = sys_cfg,
        P_t       = P_t_train,
        rng       = rng,
        log_every = gfn_cfg.log_every,
    )

    print(f"\nFinal lnZ      : {gfn.log_z.item():.4f}")
    print(f"Final avg loss : {np.mean(losses[-1000:]):.6f}")

    # ── GFlowNet vs Random comparison ───────────────────────────
    print(f"\n{'='*60}")
    print(f"  GFlowNet vs Random Scheduling (500 trials)")
    print(f"{'='*60}")

    gfn_rewards    = []
    random_rewards = []
    rng_cmp        = np.random.default_rng(1)

    for _ in range(500):
        ch = generate_channels_mmwave(
            N_t             = sys_cfg.N_t,
            N               = sys_cfg.N,
            kappa           = sys_cfg.kappa,
            L_p             = sys_cfg.L_p,
            sigma_alpha     = sys_cfg.sigma_alpha,
            theta_E_min_deg = sys_cfg.theta_E_min_deg,
            theta_E_max_deg = sys_cfg.theta_E_max_deg,
            seed            = int(rng_cmp.integers(0, 2**31)),
        )
        beta = sys_cfg.beta_mag * np.exp(1j*rng_cmp.uniform(0, 2*np.pi))
        sensing_state = run_sensing(
            theta_E_deg = ch["theta_E"],
            beta        = beta,
            N_t         = sys_cfg.N_t,
            N_r         = sys_cfg.N_r,
            L           = sys_cfg.L,
            P0          = P_t_train,
            sigma2_R    = sys_cfg.sigma2_R,
            seed        = int(rng_cmp.integers(0, 2**31)),
        )

        user_feats_np, eve_ctx_np = build_features(
            ch["H"], sensing_state, sys_cfg.N
        )

        # GFlowNet
        selected_gfn = gfn.sample(user_feats_np, eve_ctx_np, greedy=True)
        r_gfn = compute_reward(
            selected_gfn, ch["H"],
            sensing_state["theta_hat"], sensing_state["crb"],
            P_t_train, sys_cfg, rng_cmp, gfn_cfg.n_mc,
            an_type = "directed",
            g_e_hat = sensing_state["g_e_hat"],
        )
        gfn_rewards.append(r_gfn)

        # Random
        selected_rand = tuple(
            rng_cmp.choice(sys_cfg.N, sys_cfg.K, replace=False)
        )
        r_rand = compute_reward(
            selected_rand, ch["H"],
            sensing_state["theta_hat"], sensing_state["crb"],
            P_t_train, sys_cfg, rng_cmp, gfn_cfg.n_mc,
            an_type = "directed",
            g_e_hat = sensing_state["g_e_hat"],
        )
        random_rewards.append(r_rand)

    gfn_arr  = np.array(gfn_rewards)
    rand_arr = np.array(random_rewards)

    print(f"  GFlowNet mean : {gfn_arr.mean():.4f} bits/s/Hz")
    print(f"  Random mean   : {rand_arr.mean():.4f} bits/s/Hz")
    print(f"  Gain          : {gfn_arr.mean()/rand_arr.mean():.4f}x")
    print(f"  Beats random  : "
          f"{100*np.mean(gfn_arr > rand_arr):.1f}% of trials")