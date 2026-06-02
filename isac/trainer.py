"""Training pipeline for ISAC DL scheduling.

Loss function (paper eq. loss):
    L = -1/N * sum_k [lambda+ * y_k * log(p_hat_k)
                    + lambda- * (1-y_k) * log(1-p_hat_k)]
      + alpha * (sum_k p_hat_k - Kd)^2

    where:
        lambda+ = (N - Kd) / Kd   (upweights minority scheduled class)
        lambda- = 1
        alpha                     (cardinality regularizer weight)
        p_hat_k = sigmoid(logit_k)

Metric:
    Top-k accuracy — fraction of oracle-selected users correctly predicted.

Trainer:
    Adam + CosineAnnealingLR, validation each epoch, best checkpoint saved.

Reference:
    SecureLEO lab code (trainer.py) — adapted for terrestrial ISAC.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import SystemConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss function  (paper eq. loss)
# ---------------------------------------------------------------------------

def bce_topk_loss(
    logits:      Tensor,
    labels:      Tensor,
    Kd:          int,
    card_weight: float = 0.1,
) -> Tensor:
    """Weighted BCE + cardinality regularizer (paper eq. loss).

    L = BCE_weighted + alpha * (sum_k sigmoid(logit_k) - Kd)^2

    BCE weights:
        lambda+ = (N - Kd) / Kd   upweights positive (scheduled) class
        lambda- = 1

    Cardinality regularizer:
        Penalises deviations from exactly Kd scheduled users.
        alpha = card_weight (default 0.1 from paper).

    Args:
        logits      : (batch, N) raw scheduling scores
        labels      : (batch, N) binary oracle mask
        Kd          : number of users to schedule
        card_weight : alpha, cardinality regularizer weight
    Returns:
        scalar loss
    """
    N  = logits.shape[1]
    lp = (N - Kd) / Kd    # lambda+
    lm = 1.0               # lambda-

    # Weighted BCE
    weight = (1.0 - labels) * lm + labels * lp
    bce    = F.binary_cross_entropy_with_logits(
        logits, labels, weight=weight, reduction="mean"
    )

    # Cardinality regularizer: (sum_k p_hat_k - Kd)^2
    p_hat  = torch.sigmoid(logits)               # (batch, N)
    card   = (p_hat.sum(dim=1) - Kd) ** 2        # (batch,)
    reg    = card_weight * card.mean()

    return bce + reg


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def topk_accuracy(logits: Tensor, labels: Tensor, k: int) -> float:
    """Fraction of oracle-selected users correctly predicted in top-k.

    correct_i = |pred_topk_i ∩ oracle_topk_i| / k

    Args:
        logits : (batch, N) scheduling scores
        labels : (batch, N) binary oracle mask
        k      : Kd
    Returns:
        mean accuracy in [0, 1]
    """
    _, pred_idx = torch.topk(logits, k, dim=1)
    _, true_idx = torch.topk(labels, k, dim=1)
    correct = 0
    for i in range(logits.shape[0]):
        pred     = set(pred_idx[i].cpu().tolist())
        true     = set(true_idx[i].cpu().tolist())
        correct += len(pred & true)
    return correct / (logits.shape[0] * k)


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    best_val_loss:     float
    best_val_accuracy: float
    best_epoch:        int
    train_losses:      list[float]
    val_losses:        list[float]
    val_accuracies:    list[float]
    model_path:        Path | None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Training loop with validation, checkpointing, and LR scheduling.

    Args:
        model        : SetTransformerScheduler or DeepSetsScheduler
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        cfg          : SystemConfig  (all hyperparameters)
        Kd           : number of users to schedule
        device       : torch device  (None = auto-detect)
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          SystemConfig,
        Kd:           int,
        device:       torch.device | None = None,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.Kd           = Kd
        self.card_weight  = cfg.card_weight

        self.device = device or _resolve_device(cfg.device)
        self.model.to(self.device)

        self.optimizer = Adam(
            model.parameters(),
            lr           = cfg.learning_rate,
            weight_decay = cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max   = cfg.num_epochs,
            eta_min = cfg.learning_rate * 0.01,
        )

        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_acc  = 0.0
        self.best_epoch    = 0

    # ----------------------------------------------------------------
    # Public
    # ----------------------------------------------------------------

    def train(self, model_name: str = "isac_scheduler") -> TrainingResult:
        """Run full training loop.

        Args:
            model_name : prefix for checkpoint filenames
        Returns:
            TrainingResult with losses, accuracies, best checkpoint path
        """
        train_losses, val_losses, val_accs = [], [], []

        print(f"  Device      : {self.device}")
        print(f"  Epochs      : {self.cfg.num_epochs}")
        print(f"  Kd          : {self.Kd}")
        print(f"  card_weight : {self.card_weight}")
        print()

        for epoch in range(self.cfg.num_epochs):
            self.current_epoch = epoch
            t_loss             = self._train_epoch()
            v_loss, v_acc      = self._validate()
            self.scheduler.step()

            train_losses.append(t_loss)
            val_losses.append(v_loss)
            val_accs.append(v_acc)

            lr = self.scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch+1:>3}/{self.cfg.num_epochs} | "
                f"Train: {t_loss:.4f} | Val: {v_loss:.4f} | "
                f"Acc: {v_acc:.4f} | LR: {lr:.2e}"
            )

            if v_loss < self.best_val_loss:
                self.best_val_loss = v_loss
                self.best_val_acc  = v_acc
                self.best_epoch    = epoch
                self._save(self.cfg.checkpoint_dir / f"{model_name}_best.pt")

        final_path = self.cfg.checkpoint_dir / f"{model_name}_final.pt"
        self._save(final_path)
        print(
            f"\n  Best epoch : {self.best_epoch+1}  "
            f"val_loss={self.best_val_loss:.4f}  "
            f"acc={self.best_val_acc:.4f}"
        )

        return TrainingResult(
            best_val_loss     = self.best_val_loss,
            best_val_accuracy = self.best_val_acc,
            best_epoch        = self.best_epoch,
            train_losses      = train_losses,
            val_losses        = val_losses,
            val_accuracies    = val_accs,
            model_path        = self.cfg.checkpoint_dir / f"{model_name}_best.pt",
        )

    # ----------------------------------------------------------------
    # Private
    # ----------------------------------------------------------------

    def _train_epoch(self) -> float:
        self.model.train()
        total, n = 0.0, 0
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch+1}",
            leave=False,
        )
        for local, global_, labels in pbar:
            local   = local.to(self.device)
            global_ = global_.to(self.device)
            labels  = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(local, global_)
            loss   = bce_topk_loss(logits, labels, self.Kd, self.card_weight)
            loss.backward()
            self.optimizer.step()

            total += loss.item()
            n     += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total / max(n, 1)

    def _validate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss, total_acc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for local, global_, labels in self.val_loader:
                local   = local.to(self.device)
                global_ = global_.to(self.device)
                labels  = labels.to(self.device)
                logits  = self.model(local, global_)
                total_loss += bce_topk_loss(
                    logits, labels, self.Kd, self.card_weight
                ).item()
                total_acc  += topk_accuracy(logits, labels, self.Kd)
                n += 1
        return total_loss / max(n, 1), total_acc / max(n, 1)

    def _save(self, path: Path) -> None:
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch":                self.current_epoch,
            "best_val_loss":        self.best_val_loss,
            "config": {
                "local_dim":  self.model.local_dim,
                "global_dim": self.model.global_dim,
                "Kd":         self.Kd,
            },
        }, path)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def _resolve_device(device_str: str) -> torch.device:
    if device_str is None or device_str.lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)
