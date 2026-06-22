#!/usr/bin/env python3
"""Main entry point for ISAC-aided PLS with DL scheduling.

Commands:
    python main.py train      -- generate dataset and train Set Transformer
    python main.py evaluate   -- evaluate trained model vs baselines
    python main.py simulate   -- run B1/B2/Oracle Monte Carlo simulation
    python main.py version    -- show versions and device info

Usage examples:
    python main.py simulate
    python main.py train
    python main.py train --arch deep_sets --epochs 20 --samples 10000
    python main.py evaluate checkpoints/isac_set_transformer_best.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import torch


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_device(device_str: str = "auto") -> "torch.device":
    if torch is None:
        raise ImportError("PyTorch is required. Install via: pip install torch")
    if device_str.lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _patch_signal():
    """Fix signal module in restricted environments."""
    import signal
    if not hasattr(signal, "SIGINT"):
        signal.SIGINT  = 2
        signal.SIGTERM = 15


# ---------------------------------------------------------------------------
# simulate command
# ---------------------------------------------------------------------------

def cmd_simulate(args: argparse.Namespace) -> None:
    """Run B1 / B2 / Oracle Monte Carlo simulation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from isac.config import SystemConfig
    from isac.simulate import simulate_rsec_vs_snr, plot_rsec_vs_snr, plot_outage_vs_snr

    cfg = SystemConfig()
    print(cfg.summary())
    print()

    snr_db, rsec_B1, rsec_B2, rsec_oracle, pout_B1, pout_B2, pout_oracle = \
        simulate_rsec_vs_snr(cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for ext in ["png", "pdf"]:
        fig = plot_rsec_vs_snr(
            snr_db, rsec_B1, rsec_B2, rsec_oracle,
            cfg=cfg,
            save_path=str(cfg.output_dir / f"sim_rsec_vs_snr.{ext}"),
        )
        plt.close(fig)

    for ext in ["png", "pdf"]:
        fig = plot_outage_vs_snr(
            snr_db, pout_B1, pout_B2, pout_oracle,
            cfg=cfg,
            save_path=str(cfg.output_dir / f"sim_outage_vs_snr.{ext}"),
        )
        plt.close(fig)

    print("\n  Done.")


# ---------------------------------------------------------------------------
# train command
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    """Generate dataset (if needed) and train DL scheduler."""
    _patch_signal()
    from isac.config import SystemConfig
    from isac.dataset import generate_training_dataset, create_data_loaders, ISACDataset
    from isac.models import SetTransformerScheduler, DeepSetsScheduler
    from isac.trainer import Trainer
    from isac.simulate import _topology_tag

    cfg = SystemConfig(
        num_epochs    = args.epochs,
        learning_rate = args.lr,
        batch_size    = args.batch_size,
        num_samples   = args.samples,
        embed_dim     = args.embed_dim,
        num_heads     = args.num_heads,
        num_layers    = args.num_layers,
        device        = args.device,
    )

    print(cfg.summary())
    device = resolve_device(cfg.device)
    print(f"\n  Device : {device}\n")

    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    # Tag dataset and checkpoint with topology
    tag       = _topology_tag(cfg)
    data_path = cfg.data_dir / \
        f"dataset_N{cfg.N}_Kd{cfg.Kd}_M{cfg.M}{tag}.npz"

    if not data_path.exists() or args.regenerate:
        print(f"  Generating {cfg.num_samples:,} training samples ...")
        generate_training_dataset(
            output_path = data_path,
            cfg         = cfg,
            seed        = cfg.seed,
        )
    else:
        print(f"  Using existing dataset: {data_path}")

    train_loader, val_loader = create_data_loaders(
        data_path,
        batch_size = cfg.batch_size,
        val_split  = cfg.val_split,
    )

    ds = ISACDataset(data_path)
    print(f"  local_dim={ds.local_dim}, global_dim={ds.global_dim}\n")

    arch = args.arch.lower()
    if arch == "set_transformer":
        model = SetTransformerScheduler(
            local_dim  = ds.local_dim,
            global_dim = ds.global_dim,
            embed_dim  = cfg.embed_dim,
            num_heads  = cfg.num_heads,
            num_layers = cfg.num_layers,
            ff_dim     = cfg.ff_dim,
            dropout    = cfg.dropout,
        )
        model_name = f"isac_set_transformer{tag}"
    else:
        model = DeepSetsScheduler(
            local_dim  = ds.local_dim,
            global_dim = ds.global_dim,
            embed_dim  = cfg.embed_dim,
            hidden_dim = cfg.ff_dim,
            dropout    = cfg.dropout,
        )
        model_name = f"isac_deep_sets{tag}"

    total = sum(p.numel() for p in model.parameters())
    print(f"  Architecture : {arch}")
    print(f"  Parameters   : {total:,}")
    print(f"  Model name   : {model_name}\n")

    trainer = Trainer(
        model, train_loader, val_loader,
        cfg, Kd=cfg.Kd, device=device
    )
    result = trainer.train(model_name=model_name)
    print(f"\n  Best checkpoint : {result.model_path}")


# ---------------------------------------------------------------------------
# evaluate command — parallel CPU worker + serial GPU inference
# ---------------------------------------------------------------------------

def _eval_trial_cpu(t, s_idx, snr, cfg, beta_s_mag):
    """CPU-bound trial: channel gen + sensing + B1/B2/Oracle."""
    import numpy as np
    from isac.channel import generate_channels
    from isac.sensing import run_sensing
    from isac.signal import compute_secrecy_sum_rate
    from isac.scheduling import random_scheduling, oracle_scheduling_genie, mask_to_indices

    seed_t = cfg.seed + t + s_idx * 100_000
    rng    = np.random.default_rng(seed_t)
    P_t    = 10 ** (snr / 10.0) * cfg.sigma2_C

    sample = generate_channels(
        M=cfg.M, N=cfg.N, d_0=cfg.d_0, d_be=cfg.d_be,
        eta=cfg.eta, d_cu_min=cfg.d_cu_min, d_cu_max=cfg.d_cu_max,
        sigma_e=cfg.sigma_e,
        theta_E_min=cfg.theta_E_min_rad,
        theta_E_max=cfg.theta_E_max_rad,
        seed=seed_t,
    )
    H, g_e, theta_E = sample["H"], sample["g_e"], sample["theta_E"]

    beta_s_t = beta_s_mag * np.exp(1j * rng.uniform(0, 2 * np.pi))
    state    = run_sensing(
        theta_E, cfg.M, cfg.M_r, cfg.L,
        cfg.P_s, cfg.sigma2_s, beta_s_t, seed=seed_t,
    )
    g_hat_e = state["beta_hat"] * state["at_hat"]

    # B1 and B2 share the same random scheduling — only AN design differs
    sched_rand = mask_to_indices(random_scheduling(cfg.N, cfg.Kd, rng))

    r_B1 = compute_secrecy_sum_rate(
        H, g_e, sched_rand, None,
        P_t, cfg.sigma2_e, cfg.sigma2_C, cfg.time_frac, cfg.rho)

    r_B2 = compute_secrecy_sum_rate(
        H, g_e, sched_rand, g_hat_e,
        P_t, cfg.sigma2_e, cfg.sigma2_C, cfg.time_frac, cfg.rho)

    _, r_ora = oracle_scheduling_genie(
        H, g_e, cfg.Kd, P_t,
        cfg.sigma2_e, cfg.sigma2_C, cfg.time_frac, cfg.rho)

    return H, g_e, g_hat_e, state, r_B1, r_B2, r_ora, P_t


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate trained model(s) against B1, B2, Oracle baselines."""
    _patch_signal()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from joblib import Parallel, delayed

    from isac.config import SystemConfig
    from isac.models import SetTransformerScheduler, DeepSetsScheduler
    from isac.sensing import compute_beta_s
    from isac.signal import compute_secrecy_sum_rate
    from isac.scheduling import mask_to_indices
    from isac.dataset import extract_local_features, extract_global_features
    from isac.simulate import plot_rsec_vs_snr, plot_outage_vs_snr, _topology_tag

    cfg    = SystemConfig()
    device = resolve_device(args.device)

    def load_model(path):
        ckpt       = torch.load(path, map_location=device)
        mc         = ckpt.get("config", {})
        local_dim  = mc.get("local_dim",  cfg.local_dim)
        global_dim = mc.get("global_dim", cfg.global_dim)

        try:
            m = SetTransformerScheduler(
                local_dim=local_dim, global_dim=global_dim,
                embed_dim=cfg.embed_dim, num_heads=cfg.num_heads,
                num_layers=cfg.num_layers,
            )
            m.load_state_dict(ckpt["model_state_dict"])
            m.to(device).eval()
            return m, "SetTransformerScheduler"
        except Exception:
            pass

        try:
            m = DeepSetsScheduler(
                local_dim=local_dim, global_dim=global_dim,
                embed_dim=cfg.embed_dim, hidden_dim=cfg.ff_dim,
                dropout=cfg.dropout,
            )
            m.load_state_dict(ckpt["model_state_dict"])
            m.to(device).eval()
            return m, "DeepSetsScheduler"
        except Exception:
            pass

        raise ValueError(f"Could not load model from {path}")

    model_st, name_st = load_model(args.checkpoint)
    print(f"  Loaded {name_st} from {args.checkpoint}")

    model_ds = None
    if args.deepsets_checkpoint is not None:
        model_ds, name_ds = load_model(args.deepsets_checkpoint)
        print(f"  Loaded {name_ds} from {args.deepsets_checkpoint}")

    print(f"  Device : {device}\n")

    beta_s_mag    = compute_beta_s(cfg.d_be, cfg.f_c, cfg.epsilon_dBsm)
    snr_db        = np.linspace(cfg.snr_min_dB, cfg.snr_max_dB, cfg.n_snr_pts)
    n_trials      = args.trials

    rsec_B1       = np.zeros(len(snr_db))
    rsec_B2       = np.zeros(len(snr_db))
    rsec_oracle   = np.zeros(len(snr_db))
    rsec_proposed = np.zeros(len(snr_db))
    rsec_conv_dl  = np.zeros(len(snr_db))
    rsec_deepsets = np.zeros(len(snr_db)) if model_ds is not None else None

    pout_B1       = np.zeros(len(snr_db))
    pout_B2       = np.zeros(len(snr_db))
    pout_oracle   = np.zeros(len(snr_db))
    pout_proposed = np.zeros(len(snr_db))
    pout_conv_dl  = np.zeros(len(snr_db))

    hdr = f"  {'SNR':>6} | {'B1':>8} | {'B2':>8} | {'Oracle':>8} | {'B3(DL)':>9} | {'Proposed':>10}"
    if model_ds is not None:
        hdr += f" | {'DeepSets':>10}"
    print(hdr)
    print("  " + "-" * len(hdr.lstrip()))

    def model_sched(model, H, state, snr):
        loc = extract_local_features(H)
        glb = extract_global_features(
            state["theta_hat"], state["crb"], H, cfg.rho, snr, cfg.sigma2_C)
        with torch.no_grad():
            loc_t = torch.from_numpy(loc).unsqueeze(0).to(device)
            glb_t = torch.from_numpy(glb).unsqueeze(0).to(device)
            mask  = model.predict_topk(loc_t, glb_t, k=cfg.Kd)
        return mask_to_indices(mask.squeeze(0).cpu().numpy())

    for s, snr in enumerate(snr_db):
        P_t = 10 ** (snr / 10.0) * cfg.sigma2_C

        cpu_results = Parallel(n_jobs=-1)(
            delayed(_eval_trial_cpu)(t, s, snr, cfg, beta_s_mag)
            for t in range(n_trials)
        )

        a_B1 = a_B2 = a_ora = a_cdl = a_st = a_ds = 0.0
        o_B1 = o_B2 = o_ora = o_cdl = o_st = 0.0

        for H, g_e, g_hat_e, state, r_B1, r_B2, r_ora, P_t_t in cpu_results:
            a_B1  += r_B1;  o_B1  += float(r_B1  < cfg.R0)
            a_B2  += r_B2;  o_B2  += float(r_B2  < cfg.R0)
            a_ora += r_ora; o_ora += float(r_ora  < cfg.R0)

            # Proposed: DL scheduling + directed AN
            s_st = model_sched(model_st, H, state, snr)
            r_st = compute_secrecy_sum_rate(
                H, g_e, s_st, g_hat_e,
                P_t_t, cfg.sigma2_e, cfg.sigma2_C, cfg.time_frac, cfg.rho)
            a_st += r_st; o_st += float(r_st < cfg.R0)

            # B3: DL scheduling + isotropic AN
            r_cdl = compute_secrecy_sum_rate(
                H, g_e, s_st, None,
                P_t_t, cfg.sigma2_e, cfg.sigma2_C, cfg.time_frac, cfg.rho)
            a_cdl += r_cdl; o_cdl += float(r_cdl < cfg.R0)

            if model_ds is not None:
                s_ds  = model_sched(model_ds, H, state, snr)
                a_ds += compute_secrecy_sum_rate(
                    H, g_e, s_ds, g_hat_e,
                    P_t_t, cfg.sigma2_e, cfg.sigma2_C, cfg.time_frac, cfg.rho)

        rsec_B1[s]       = a_B1  / n_trials
        rsec_B2[s]       = a_B2  / n_trials
        rsec_oracle[s]   = a_ora / n_trials
        rsec_proposed[s] = a_st  / n_trials
        rsec_conv_dl[s]  = a_cdl / n_trials
        if rsec_deepsets is not None:
            rsec_deepsets[s] = a_ds / n_trials

        pout_B1[s]       = o_B1  / n_trials
        pout_B2[s]       = o_B2  / n_trials
        pout_oracle[s]   = o_ora / n_trials
        pout_proposed[s] = o_st  / n_trials
        pout_conv_dl[s]  = o_cdl / n_trials

        row = (f"  {snr:>+6.1f} | {rsec_B1[s]:>8.4f} | {rsec_B2[s]:>8.4f} | "
               f"{rsec_oracle[s]:>8.4f} | {rsec_conv_dl[s]:>9.4f} | "
               f"{rsec_proposed[s]:>10.4f}")
        if rsec_deepsets is not None:
            row += f" | {rsec_deepsets[s]:>10.4f}"
        print(row)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results for future replotting
    tag = _topology_tag(cfg)
    np.savez(
        cfg.output_dir / f"eval_results{tag}.npz",
        snr_db        = snr_db,
        rsec_B1       = rsec_B1,
        rsec_B2       = rsec_B2,
        rsec_oracle   = rsec_oracle,
        rsec_proposed = rsec_proposed,
        rsec_conv_dl  = rsec_conv_dl,
        rsec_deepsets = rsec_deepsets if rsec_deepsets is not None \
                        else np.array([]),
        pout_B1       = pout_B1,
        pout_B2       = pout_B2,
        pout_oracle   = pout_oracle,
        pout_proposed = pout_proposed,
        pout_conv_dl  = pout_conv_dl,
    )
    print(f"  Results saved -> eval_results{tag}.npz")

    def nan_zeros(arr):
        out = arr.copy().astype(float)
        out[out == 0.0] = np.nan
        return out

    for ext in ["png", "pdf"]:
        fig = plot_rsec_vs_snr(
            snr_db, rsec_B1, rsec_B2, rsec_oracle,
            rsec_proposed = rsec_proposed,
            rsec_deepsets = None,
            rsec_conv_dl  = rsec_conv_dl,
            cfg           = cfg,
            save_path     = str(cfg.output_dir / f"eval_rsec_vs_snr.{ext}"),
        )
        plt.close(fig)

    for ext in ["png", "pdf"]:
        fig = plot_outage_vs_snr(
            snr_db,
            nan_zeros(pout_B1),
            nan_zeros(pout_B2),
            nan_zeros(pout_oracle),
            pout_proposed = nan_zeros(pout_proposed),
            pout_conv_dl  = nan_zeros(pout_conv_dl),
            cfg           = cfg,
            save_path     = str(cfg.output_dir / f"eval_outage_vs_snr.{ext}"),
        )
        plt.close(fig)

    print("\n  Done.")


# ---------------------------------------------------------------------------
# version command
# ---------------------------------------------------------------------------

def cmd_version(args: argparse.Namespace) -> None:
    """Show version and device info."""
    _patch_signal()
    try:
        import torch
        print(f"PyTorch    : {torch.__version__}")
        print(f"CUDA       : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU        : {torch.cuda.get_device_name(0)}")
    except Exception:
        print("PyTorch    : not available")
    import numpy as np
    import matplotlib
    print(f"NumPy      : {np.__version__}")
    print(f"Matplotlib : {matplotlib.__version__}")
    print(f"Python     : {sys.version.split()[0]}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="ISAC PLS Deep Learning Scheduler",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("simulate", help="Run B1/B2/Oracle Monte Carlo simulation")

    p_train = sub.add_parser("train", help="Train the DL scheduler")
    p_train.add_argument("--arch",        default="set_transformer",
                         choices=["set_transformer", "deep_sets"])
    p_train.add_argument("--num-samples", type=int,   default=1_000_000, dest="samples")
    p_train.add_argument("--num-epochs",  type=int,   default=50,        dest="epochs")
    p_train.add_argument("--batch-size",  type=int,   default=256)
    p_train.add_argument("--embed-dim",   type=int,   default=128,       dest="embed_dim")
    p_train.add_argument("--num-heads",   type=int,   default=4,         dest="num_heads")
    p_train.add_argument("--num-layers",  type=int,   default=2,         dest="num_layers")
    p_train.add_argument("--lr",          type=float, default=1e-3)
    p_train.add_argument("--device",      default="auto")
    p_train.add_argument("--regenerate",  action="store_true")

    p_eval = sub.add_parser("evaluate", help="Evaluate trained model")
    p_eval.add_argument("checkpoint",   type=Path)
    p_eval.add_argument("--deepsets",   type=Path, default=None, dest="deepsets_checkpoint")
    p_eval.add_argument("--num-trials", type=int,  default=10000, dest="trials")
    p_eval.add_argument("--device",     default="auto")

    sub.add_parser("version", help="Show version and device info")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "simulate": cmd_simulate,
        "train":    cmd_train,
        "evaluate": cmd_evaluate,
        "version":  cmd_version,
    }
    dispatch[args.command](args)
