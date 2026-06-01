"""Train the DreamerV3-style RSSM world model (spec §6–§8).

Usage
-----
    python scripts/wm_train_dreamer.py --dataset main
    # smoke
    python scripts/wm_train_dreamer.py --dataset smoke --max-steps 200 \
        --batch-size 8 --seq-len 16 --context 4 --eval-every 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from world_model.dreamer import (
    DreamerWorldModel, WorldModelConfig, compute_loss,
    LaProp, adaptive_grad_clip, SequenceReplay, evaluate,
)


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["world_model"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Dataset name under --data-root.")
    p.add_argument("--data-root", default="data/replay")
    p.add_argument("--config", default="configs/world_model/dreamer_v3.yaml")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Defaults to checkpoints/dreamer_wm/<dataset>.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    # overrides (optional)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--context", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    args = p.parse_args()

    cfg = load_cfg(str(ROOT / args.config) if not Path(args.config).is_absolute() else args.config)
    if args.max_steps is not None: cfg["max_train_steps"] = args.max_steps
    if args.batch_size is not None: cfg["batch_size"] = args.batch_size
    if args.seq_len is not None: cfg["seq_length"] = args.seq_len
    if args.context is not None: cfg["context"] = args.context
    if args.eval_every is not None: cfg["eval_every"] = args.eval_every

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    data_dir = ROOT / args.data_root / args.dataset
    window = cfg["context"] + cfg["seq_length"]
    replay = SequenceReplay(str(data_dir), length=window, seed=args.seed)
    print(f"[data] {data_dir}: {len(replay)} steps | window={window} (context={cfg['context']})")

    wm_cfg = WorldModelConfig(
        action_dim=cfg["action_dim"], groups=cfg["groups"], classes=cfg["classes"],
        deter=cfg["deter"], hidden=cfg["hidden"], e_dim=cfg["e_dim"],
        action_emb=cfg["action_emb"], num_bins=cfg["num_bins"],
        vmin=cfg["vmin"], vmax=cfg["vmax"], unimix=cfg["unimix"],
    )
    model = DreamerWorldModel(wm_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] DreamerWorldModel: {n_params/1e6:.2f}M params on {device}")

    opt = LaProp(model.parameters(), lr=cfg["learning_rate"], betas=tuple(cfg["betas"]))
    reward_bins = model.reward_head.bins

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else ROOT / "checkpoints" / "dreamer_wm" / args.dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_metric = float("inf")
    for step in range(1, cfg["max_train_steps"] + 1):
        model.train()
        batch = replay.sample_batch(cfg["batch_size"], device=device)
        outputs = model.observe(batch["states"], batch["actions"], batch["is_first"])
        loss, metrics = compute_loss(
            outputs, batch, reward_bins,
            beta_pred=cfg["beta_pred"], beta_dyn=cfg["beta_dyn"], beta_rep=cfg["beta_rep"],
            free_nats=cfg["free_nats"], context=cfg["context"],
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        adaptive_grad_clip(model.parameters(), clip=cfg["agc_clip"])
        opt.step()

        if step % cfg["log_every"] == 0 or step == 1:
            print(f"step {step:6d} | loss {metrics['loss']:.4f} "
                  f"| pred {metrics['L_pred']:.4f} dyn {metrics['L_dyn']:.3f} rep {metrics['L_rep']:.3f} "
                  f"| rew_mse {metrics['reward_mse']:.4f} cont_acc {metrics['cont_acc']:.3f} "
                  f"| recon(c {metrics['recon_cont']:.3f}/b {metrics['recon_bin']:.3f})")
            if not np.isfinite(metrics["loss"]):
                sys.exit("Loss diverged to NaN/Inf — aborting.")

        if step % cfg["eval_every"] == 0 or step == cfg["max_train_steps"]:
            ev = evaluate(model, replay, context=cfg["context"], horizon=cfg["k_step"],
                          n_windows=cfg["n_eval_windows"], device=device, seed=cfg["eval_seed"])
            print(f"  [eval @ {step}] "
                  f"1step rew_mse {ev['one_step/reward_mse']:.4f} "
                  f"recon_bin_acc {ev['one_step/recon_bin_acc']:.3f} "
                  f"cont_acc {ev['one_step/cont_acc']:.3f} "
                  f"| kstep rew_mse(final) {ev['kstep/reward_mse_final']:.4f} "
                  f"recon_cont_mse(final) {ev['kstep/recon_cont_mse_final']:.4f} "
                  f"| collapsed_groups {ev['collapse/n_collapsed_groups']}/{cfg['groups']} "
                  f"ent {ev['collapse/mean_group_entropy']:.3f}/{ev['collapse/max_entropy']:.3f}")
            score = ev["one_step/reward_mse"] + ev["kstep/recon_cont_mse_final"]
            if score < best_metric:
                best_metric = score
                torch.save({"model": model.state_dict(), "cfg": vars(wm_cfg), "step": step},
                           ckpt_dir / "best.pt")
            torch.save({"model": model.state_dict(), "cfg": vars(wm_cfg), "step": step},
                       ckpt_dir / "latest.pt")

    print(f"Done. Checkpoints in {ckpt_dir} (best score {best_metric:.4f})")


if __name__ == "__main__":
    main()
