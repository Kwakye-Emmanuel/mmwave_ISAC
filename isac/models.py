"""Neural network models for ISAC-aided DL scheduling.

Set Transformer scheduler:
    Input:  local  (batch, N, local_dim)   -- per-user channel features
            global_ (batch, global_dim)    -- sensing output + SNR
    Output: logits (batch, N)              -- scheduling scores per user
    Select: top-Kd users by logit score

Architecture:
    local -> Linear(embed_dim) -> SAB x num_layers -> PMA pool
    [z_n, pooled, global_] -> Linear -> logit_n

Global feature vector (Option A, global_dim=6):
    v^global = [theta_hat, crb, mu, sigma_h^2, rho, snr_lin]^T

Local feature vector (local_dim = 2*M):
    v_n^local = [Re(h_n)^T, Im(h_n)^T]^T

References:
    Set Transformer : Lee et al., ICML 2019
    Deep Sets       : Zaheer et al., NeurIPS 2017
    SecureLEO lab code (models.py) -- adapted for terrestrial ISAC
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Set Transformer building blocks
# ---------------------------------------------------------------------------

class MultiheadAttentionBlock(nn.Module):
    """MAB(X, Y) = LayerNorm(H + FFN(H)) where H = LayerNorm(X + Attn(X,Y,Y))."""

    def __init__(
        self,
        dim:     int,
        heads:   int   = 4,
        ff_dim:  int   = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        h = self.norm1(x + self.attn(x, y, y, need_weights=False)[0])
        return self.norm2(h + self.ffn(h))


class SetAttentionBlock(nn.Module):
    """SAB(X) = MAB(X, X) — self-attention among set elements."""

    def __init__(
        self,
        dim:     int,
        heads:   int   = 4,
        ff_dim:  int   = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mab = MultiheadAttentionBlock(dim, heads, ff_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.mab(x, x)


class PoolingByMultiheadAttention(nn.Module):
    """PMA: learnable seed vectors attend to set elements."""

    def __init__(
        self,
        dim:       int,
        num_seeds: int   = 1,
        heads:     int   = 4,
        ff_dim:    int   = 256,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(num_seeds, dim))
        self.mab   = MultiheadAttentionBlock(dim, heads, ff_dim, dropout)

    def forward(self, z: Tensor) -> Tensor:
        seeds = self.seeds.unsqueeze(0).expand(z.shape[0], -1, -1)
        return self.mab(seeds, z)


# ---------------------------------------------------------------------------
# Set Transformer Scheduler  (proposed method)
# ---------------------------------------------------------------------------

class SetTransformerScheduler(nn.Module):
    """Set Transformer for ISAC user scheduling.

    Architecture:
        local  -> Linear(embed_dim) -> ReLU  [input embed]
               -> SAB x num_layers            [inter-user attention]
               -> PMA                         [global pool]
        [z_n, pooled, global_] -> Linear -> logit_n

    Permutation invariant: output logit_n depends on all users equally,
    not on their ordering in the input.

    Args:
        local_dim  : per-user feature dimension  (= 2*M)
        global_dim : global feature dimension    (= 6)
        embed_dim  : transformer embedding size  (default 128)
        num_heads  : attention heads             (default 4)
        num_layers : number of SAB layers        (default 2)
        ff_dim     : feedforward hidden size     (default 256)
        dropout    : dropout rate                (default 0.1)
    """

    def __init__(
        self,
        local_dim:    int,
        global_dim:   int,
        embed_dim:    int   = 128,
        num_heads:    int   = 4,
        num_layers:   int   = 2,
        ff_dim:       int   = 256,
        num_pma_seeds: int  = 1,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.local_dim  = local_dim
        self.global_dim = global_dim
        self.embed_dim  = embed_dim

        self.input_embed = nn.Sequential(
            nn.Linear(local_dim, embed_dim), nn.ReLU()
        )
        self.encoder = nn.ModuleList([
            SetAttentionBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.pma  = PoolingByMultiheadAttention(
            embed_dim, num_pma_seeds, num_heads, ff_dim, dropout
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim + embed_dim + global_dim, ff_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, 1),
        )

    def forward(self, local: Tensor, global_: Tensor) -> Tensor:
        """Forward pass.

        Args:
            local   : (batch, N, local_dim)   per-user features
            global_ : (batch, global_dim)     sensing output + SNR
        Returns:
            logits  : (batch, N)              scheduling score per user
        """
        B, N, _ = local.shape

        # Embed each user's features
        z = self.input_embed(local)               # (B, N, embed_dim)

        # Inter-user attention
        for sab in self.encoder:
            z = sab(z)                            # (B, N, embed_dim)

        # Global pool via PMA
        pooled = self.pma(z).mean(dim=1, keepdim=True).expand(B, N, self.embed_dim)

        # Broadcast global features to each user
        g = global_.unsqueeze(1).expand(B, N, self.global_dim)

        # Per-user decision head
        return self.head(torch.cat([z, pooled, g], dim=-1)).squeeze(-1)

    def predict_topk(self, local: Tensor, global_: Tensor, k: int) -> Tensor:
        """Return binary scheduling mask selecting top-k users.

        Args:
            local   : (batch, N, local_dim)
            global_ : (batch, global_dim)
            k       : number of users to select (= Kd)
        Returns:
            mask : (batch, N) binary float tensor
        """
        logits = self.forward(local, global_)
        _, idx = torch.topk(logits, k, dim=1)
        mask   = torch.zeros_like(logits)
        mask.scatter_(1, idx, 1.0)
        return mask


# ---------------------------------------------------------------------------
# Deep Sets Scheduler  (baseline comparison)
# ---------------------------------------------------------------------------

class DeepSetsScheduler(nn.Module):
    """Deep Sets baseline for ISAC user scheduling.

    Architecture:
        local  -> MLP encode -> mean pool
        [emb_n, pooled, global_] -> MLP -> logit_n

    No attention mechanism — cannot capture inter-user interactions.
    Used to show that attention is essential for good scheduling.

    Args:
        local_dim  : per-user feature dimension  (= 2*M)
        global_dim : global feature dimension    (= 6)
        embed_dim  : embedding size              (default 128)
        hidden_dim : MLP hidden size             (default 256)
        dropout    : dropout rate                (default 0.1)
    """

    def __init__(
        self,
        local_dim:  int,
        global_dim: int,
        embed_dim:  int   = 128,
        hidden_dim: int   = 256,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.local_dim  = local_dim
        self.global_dim = global_dim
        self.embed_dim  = embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(local_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim + embed_dim + global_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, local: Tensor, global_: Tensor) -> Tensor:
        """Forward pass.

        Args:
            local   : (batch, N, local_dim)
            global_ : (batch, global_dim)
        Returns:
            logits  : (batch, N)
        """
        B, N, L = local.shape
        emb    = self.encoder(local.view(B * N, L)).view(B, N, self.embed_dim)
        pooled = emb.mean(dim=1, keepdim=True).expand(B, N, self.embed_dim)
        g      = global_.unsqueeze(1).expand(B, N, self.global_dim)
        return self.head(torch.cat([emb, pooled, g], dim=-1)).squeeze(-1)

    def predict_topk(self, local: Tensor, global_: Tensor, k: int) -> Tensor:
        """Return binary scheduling mask selecting top-k users."""
        logits = self.forward(local, global_)
        _, idx = torch.topk(logits, k, dim=1)
        mask   = torch.zeros_like(logits)
        mask.scatter_(1, idx, 1.0)
        return mask