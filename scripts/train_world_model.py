"""Phase 2: Train the ensemble world model on collected transition data."""
import argparse
import sys
from pathlib import Path

import torch
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world_model import EnsembleWorldModel, SequenceReplayBuffer, evaluate_k_step_rollout
from world_model.loss import compute_world_model_loss


def train_one_step(ensemble, optimizers, train_buffer, cfg, global_step):
    total = {}
    for k, member in enumerate(ensemble.members):
        bootstrap_seed = global_step * cfg.num_ensemble_members + k
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

        outputs = member.forward_sequence(states, actions, burnin=cfg.burnin)
        loss, log_dict = compute_world_model_loss(
            outputs, rewards, dones,
            beta_reward=cfg.beta_reward,
            beta_done=cfg.beta_done,
            beta_var=cfg.beta_var,
        )

        optimizers[k].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(member.parameters(), cfg.grad_clip)
        optimizers[k].step()

        for key, val in log_dict.items():
            total[f"member_{k}/{key}"] = val
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/replay")
    parser.add_argument("--config", default="configs/world_model/jepa_default.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)["world_model"]

    class Cfg:
        pass

    cfg = Cfg()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    device = torch.device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)

    ensemble = EnsembleWorldModel(num_members=cfg.num_ensemble_members).to(device)
    optimizers = [Adam(m.parameters(), lr=cfg.learning_rate) for m in ensemble.members]

    train_buffer = SequenceReplayBuffer(args.data_dir, split="train")
    val_buffer   = SequenceReplayBuffer(args.data_dir, split="val")

    if args.wandb:
        import wandb
        wandb.init(project="cs377-team4", config=cfg_dict)

    best_val_score = float("inf")
    patience_counter = 0

    for step in range(cfg.max_train_steps):
        loss_logs = train_one_step(ensemble, optimizers, train_buffer, cfg, step)

        if step % 100 == 0:
            avg_loss = loss_logs.get("member_0/L_total", 0)
            print(f"[step {step:>6}] L_total={avg_loss:.4f}")
            if args.wandb:
                import wandb
                wandb.log({**loss_logs, "step": step})

        if step % cfg.eval_every == 0 and step > 0:
            ensemble.eval()
            val_metrics = evaluate_k_step_rollout(
                ensemble, val_buffer, K=cfg.k_step, N=cfg.n_eval_trajectories,
            )
            ensemble.train()
            print(f"[eval  {step:>6}] latent_mse={val_metrics['k_step_latent_mse']:.4f}  "
                  f"reward_mse={val_metrics['k_step_reward_mse']:.4f}  "
                  f"done_err={val_metrics['k_step_done_err']:.4f}  "
                  f"sigma={val_metrics['sigma_mean']:.4f}")
            if args.wandb:
                import wandb
                wandb.log({**{f"eval/{k}": v for k, v in val_metrics.items()}, "step": step})

            score = val_metrics["k_step_latent_mse"]
            if score < best_val_score:
                best_val_score = score
                patience_counter = 0
                ensemble.save(str(ckpt_dir / "best.pt"))
                print(f"  -> New best: {best_val_score:.4f} — saved best.pt")
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


if __name__ == "__main__":
    main()
