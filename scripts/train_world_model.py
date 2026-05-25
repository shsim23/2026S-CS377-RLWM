"""Train the (single or multi-member) world model.

Two modes:

1.  Fresh training (default — single model now that ensemble is disabled):
        python scripts/train_world_model.py --data-dir data/replay/pacman_classic --wandb

2.  Head fine-tuning from an existing checkpoint
    (encoder + dynamics + action_embedder frozen, only reward/done heads optimized):
        python scripts/train_world_model.py \
            --data-dir data/replay/pacman_classic \
            --resume-from checkpoints/pacman_classic/best.pt \
            --extract-member 0 \
            --freeze-dynamics \
            --beta-reward 3.0 --beta-done 3.0 \
            --max-train-steps 3000 --eval-every 200 \
            --wandb --wandb-name head_finetune

Notes
-----
* Train-time burn-in is sampled per step in [burnin_min, burnin_max] so the
  model sees both cold (h=0) and warm-started conditions.
* Eval uses a fixed seed for val trajectory sampling so the metric is
  comparable across steps within a single run.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world_model import EnsembleWorldModel, SequenceReplayBuffer, evaluate_k_step_rollout
from world_model.loss import compute_world_model_loss


# --------------------------------------------------------------------------- #
def train_one_step(ensemble, optimizers, train_buffer, cfg, global_step):
    """One optimisation step over every (single or multiple) ensemble member."""
    burnin = random.randint(cfg.burnin_min, cfg.burnin_max)
    total = {}
    for k, member in enumerate(ensemble.members):
        bootstrap_seed = global_step * len(ensemble.members) + k
        batch = train_buffer.sample_sequence(
            batch_size=cfg.batch_size,
            seq_length=cfg.seq_length,
            bootstrap_seed=bootstrap_seed,
        )
        device = next(member.parameters()).device
        states  = batch["states"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device)
        dones   = batch["dones"].to(device)

        outputs = member.forward_sequence(states, actions, burnin=burnin)
        loss, log_dict = compute_world_model_loss(
            outputs, rewards, dones,
            states=states,
            beta_reward=cfg.beta_reward,
            beta_done=cfg.beta_done,
            beta_var=cfg.beta_var,
            beta_dynamic_state=getattr(cfg, "beta_dynamic_state", 0.0),
            beta_count_delta=getattr(cfg, "beta_count_delta", 0.0),
            beta_food_eaten=getattr(cfg, "beta_food_eaten", 0.0),
            pos_weight_done=getattr(cfg, "pos_weight_done", 1.0),
            target_std=getattr(cfg, "target_std", 1.0),
        )

        optimizers[k].zero_grad()
        loss.backward()
        # Only clip params that have grads (head-only mode skips frozen modules)
        params_w_grad = [p for p in member.parameters() if p.grad is not None]
        if params_w_grad:
            torch.nn.utils.clip_grad_norm_(params_w_grad, cfg.grad_clip)
        optimizers[k].step()

        for key, val in log_dict.items():
            total[f"member_{k}/{key}"] = val
    total["train/burnin"] = burnin
    return total


# --------------------------------------------------------------------------- #
def _extract_single_member_state(full_sd: dict, member_idx: int) -> dict:
    """Pull one member's weights out of a multi-member state_dict, re-keyed
    as `members.0.*` so it loads into a fresh K=1 EnsembleWorldModel."""
    src_prefix = f"members.{member_idx}."
    dst_prefix = "members.0."
    out = {}
    for k, v in full_sd.items():
        if k.startswith(src_prefix):
            out[dst_prefix + k[len(src_prefix):]] = v
    return out


def load_checkpoint(ckpt_path: str, num_members: int, extract_member: int | None, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_K = ckpt["K"]
    if extract_member is not None:
        if extract_member >= ckpt_K:
            raise ValueError(f"--extract-member {extract_member} but checkpoint has K={ckpt_K}")
        sd = _extract_single_member_state(ckpt["state_dict"], extract_member)
        model = EnsembleWorldModel(num_members=1).to(device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[resume] Extracted member {extract_member} from {ckpt_path} (K={ckpt_K} -> 1)")
    else:
        if num_members != ckpt_K:
            raise ValueError(
                f"--num-members={num_members} does not match checkpoint K={ckpt_K}. "
                f"Pass --extract-member <idx> to extract one, or set --num-members {ckpt_K}."
            )
        model = EnsembleWorldModel(num_members=ckpt_K).to(device)
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"[resume] Loaded full ensemble (K={ckpt_K}) from {ckpt_path}")
    if missing:
        print(f"[resume] {len(missing)} missing keys (fresh init): "
              + ", ".join(sorted({k.split('.')[2] for k in missing if k.startswith('members.')})))
    if unexpected:
        print(f"[resume] {len(unexpected)} unexpected keys (dropped): "
              + ", ".join(sorted({k.split('.')[2] for k in unexpected if k.startswith('members.')})))
    return model


@torch.no_grad()
def compute_latent_health(ensemble: EnsembleWorldModel,
                          val_buffer: SequenceReplayBuffer,
                          n_trajs: int = 50, seed: int = 0) -> dict:
    """Sample some val states, encode, report per-dim std stats.
    Used to track latent collapse / dead dims across training.
    """
    device = next(ensemble.parameters()).device
    trajs = val_buffer.sample_trajectories(n_trajs, seed=seed)
    zs = []
    for traj in trajs:
        s = torch.from_numpy(traj["states"]).float().to(device)
        z, _ = ensemble.encode(s)
        zs.append(z.cpu().numpy())
    Z = np.concatenate(zs, axis=0)
    stds = Z.std(axis=0)
    return {
        "latent/std_mean":   float(stds.mean()),
        "latent/std_min":    float(stds.min()),
        "latent/std_max":    float(stds.max()),
        "latent/dead_dims":  int((stds < 0.01).sum()),
    }


def freeze_dynamics(ensemble: EnsembleWorldModel):
    """Freeze encoder/action_embedder/dynamics/dynamic_state_head (v8.2 backbone).
    Leave food_eaten_head + done_head trainable.  v10b: tests whether the
    v8.2-trained latent already contains enough info for FoodEatenHead to
    reach the reward threshold, without further perturbing the encoder."""
    frozen, trainable = 0, 0
    for m in ensemble.members:
        for module in (m.encoder, m.action_embedder, m.dynamics, m.dynamic_state_head):
            for p in module.parameters():
                p.requires_grad_(False)
                frozen += p.numel()
        for module in (m.food_eaten_head, m.done_head):
            for p in module.parameters():
                p.requires_grad_(True)
                trainable += p.numel()
    print(f"[freeze] frozen params: {frozen:,}   trainable (heads only): {trainable:,}")


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/replay")
    parser.add_argument("--config", default="configs/world_model/jepa_default.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--num-members", type=int, default=None,
                        help="Override num_ensemble_members from config (default 1).")
    parser.add_argument("--resume-from", default=None, help="Checkpoint .pt to resume from.")
    parser.add_argument("--extract-member", type=int, default=None,
                        help="When resuming from a multi-member ckpt, keep only this member.")
    parser.add_argument("--freeze-dynamics", action="store_true",
                        help="Freeze encoder/dynamics/action_embedder; train only reward+done heads.")

    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--burnin-min", type=int, default=None)
    parser.add_argument("--burnin-max", type=int, default=None)
    parser.add_argument("--beta-reward", type=float, default=None)
    parser.add_argument("--beta-done", type=float, default=None)
    parser.add_argument("--pos-weight-done", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)

    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for torch / numpy / python random (training reproducibility).")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-tags", default=None,
                        help="comma-separated tags for wandb run")
    args = parser.parse_args()

    # Reproducibility — Note: ensemble member init is also seeded inside
    # EnsembleWorldModel (`ensemble.py:18`), so different K give different model
    # seeds. This flag controls *the rest*: data sampling, burnin draws, etc.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    import yaml
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)["world_model"]

    class Cfg:
        pass
    cfg = Cfg()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    # CLI overrides
    overrides = {
        "max_train_steps": args.max_train_steps,
        "eval_every":      args.eval_every,
        "burnin_min":      args.burnin_min,
        "burnin_max":      args.burnin_max,
        "beta_reward":     args.beta_reward,
        "beta_done":       args.beta_done,
        "pos_weight_done": args.pos_weight_done,
        "learning_rate":   args.learning_rate,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
            cfg_dict[k] = v

    num_members = args.num_members if args.num_members is not None \
        else getattr(cfg, "num_ensemble_members", 1)

    if cfg.burnin_min > cfg.burnin_max:
        raise ValueError(f"burnin_min ({cfg.burnin_min}) > burnin_max ({cfg.burnin_max})")

    device = torch.device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- model ----
    if args.resume_from is not None:
        ensemble = load_checkpoint(args.resume_from, num_members, args.extract_member, device)
    else:
        ensemble = EnsembleWorldModel(num_members=num_members).to(device)
        print(f"[init] fresh model with K={num_members}")

    if args.freeze_dynamics:
        freeze_dynamics(ensemble)

    # one optimizer per member, over its currently trainable params
    optimizers = []
    for m in ensemble.members:
        trainable = [p for p in m.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters! Did you freeze everything?")
        optimizers.append(Adam(trainable, lr=cfg.learning_rate))

    train_buffer = SequenceReplayBuffer(args.data_dir, split="train")
    val_buffer   = SequenceReplayBuffer(args.data_dir, split="val")

    # ---- wandb ----
    run = None
    if args.wandb:
        import wandb
        run_cfg = dict(cfg_dict)
        run_cfg.update({
            "num_members": num_members,
            "freeze_dynamics": args.freeze_dynamics,
            "resume_from": args.resume_from,
            "extract_member": args.extract_member,
            "data_dir": args.data_dir,
        })
        tags = [t.strip() for t in args.wandb_tags.split(",")] if args.wandb_tags else None
        run = wandb.init(
            project=getattr(cfg, "wandb_project", "cs377-team4"),
            entity=getattr(cfg, "wandb_entity", None),
            name=args.wandb_name,
            tags=tags,
            config=run_cfg,
        )

    best_val_score = float("inf")
    patience_counter = 0
    eval_seed = getattr(cfg, "eval_seed", 0)

    # --------------------------------------------------------------------- #
    for step in range(cfg.max_train_steps):
        loss_logs = train_one_step(ensemble, optimizers, train_buffer, cfg, step)

        if step % 100 == 0:
            avg_loss = loss_logs.get("member_0/L_total", 0)
            burnin_used = loss_logs.get("train/burnin", -1)
            print(f"[step {step:>6}] L_total={avg_loss:.4f}  burnin={burnin_used}")

        if run is not None:
            # log every step so wandb gets a dense curve
            run.log({**loss_logs, "step": step})

        if step > 0 and step % cfg.eval_every == 0:
            ensemble.eval()
            val_metrics = evaluate_k_step_rollout(
                ensemble, val_buffer,
                K=cfg.k_step, N=cfg.n_eval_trajectories, seed=eval_seed,
            )
            latent_health = compute_latent_health(
                ensemble, val_buffer, n_trajs=50, seed=eval_seed,
            )
            ensemble.train()
            print(f"[eval  {step:>6}] "
                  f"latent_mse={val_metrics['k_step_latent_mse']:.4f}  "
                  f"reward_mse={val_metrics['k_step_reward_mse']:.4f}  "
                  f"done_err={val_metrics['k_step_done_err']:.4f}  "
                  f"sigma={val_metrics['sigma_mean']:.4f}  "
                  f"dead_dims={latent_health['latent/dead_dims']}")
            if run is not None:
                run.log({**{f"eval/{k}": v for k, v in val_metrics.items()},
                         **latent_health, "step": step})

            # combined score: weighted sum so best.pt reflects all three heads
            score = (val_metrics["k_step_latent_mse"]
                     + val_metrics["k_step_reward_mse"]
                     + val_metrics["k_step_done_err"])
            if run is not None:
                run.log({"eval/combined_score": score, "step": step})

            if score < best_val_score:
                best_val_score = score
                patience_counter = 0
                ensemble.save(str(ckpt_dir / "best.pt"))
                print(f"  -> New best (combined): {best_val_score:.4f} — saved best.pt")
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"Early stopping at step {step}")
                    break

            if (val_metrics["k_step_latent_mse"] < cfg.threshold_latent_mse and
                    val_metrics["k_step_reward_mse"] < cfg.threshold_reward_mse and
                    val_metrics["k_step_done_err"]   < cfg.threshold_done_err):
                print(f"All thresholds met at step {step}")
                ensemble.save(str(ckpt_dir / "best.pt"))
                break

    ensemble.save(str(ckpt_dir / "final.pt"))
    print("Training complete.")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
