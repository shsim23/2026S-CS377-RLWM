"""End-to-end pipeline: collect data for one layout, then train its world model.

Example
-------
# Full run on the pacman_classic layout
python scripts/train_pipeline.py --layout-name pacman_classic

# Quick smoke (small data + few train steps)
python scripts/train_pipeline.py --layout-name pacman_classic \
    --num-transitions 3000 --max-train-steps 200 --eval-every 100

# Re-train only (skip data collection if data/replay/<layout>/train already exists)
python scripts/train_pipeline.py --layout-name pacman_classic --skip-collect
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-name", required=True,
                        help="Layout basename under layouts/train/ (without .txt).")
    # Data collection
    parser.add_argument("--num-transitions", type=int, default=70000)
    parser.add_argument("--policy", default="mixed", choices=["random", "mixed"])
    parser.add_argument("--p-greedy", type=float, default=0.3)
    parser.add_argument("--ghost-epsilon", type=float, default=0.2,
                        help="Design-frozen default; keep at 0.2 to preserve VPA narrative.")
    parser.add_argument("--num-ghosts", type=int, default=1)
    parser.add_argument("--randomize-spawn", dest="randomize_spawn",
                        action="store_true", default=True)
    parser.add_argument("--no-randomize-spawn", dest="randomize_spawn",
                        action="store_false")
    parser.add_argument("--min-spawn-dist", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--collect-seed", type=int, default=0)
    parser.add_argument("--skip-collect", action="store_true",
                        help="Skip collection if data/replay/<layout>/train already exists.")
    parser.add_argument("--force-collect", action="store_true",
                        help="Wipe existing data/replay/<layout>/ before collecting.")
    # Training
    parser.add_argument("--config", default="configs/world_model/jepa_default.yaml")
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--checkpoint-subdir", default=None,
                        help="checkpoints/<this>/. Defaults to layout-name.")
    args = parser.parse_args()

    layout_path = ROOT / "layouts" / "train" / f"{args.layout_name}.txt"
    if not layout_path.exists():
        sys.exit(f"Layout not found: {layout_path}")

    data_dir = ROOT / "data" / "replay" / args.layout_name
    ckpt_dir = ROOT / "checkpoints" / (args.checkpoint_subdir or args.layout_name)

    # -------- Phase 1: collect data --------
    train_dir_exists = (data_dir / "train").exists() and any((data_dir / "train").glob("*.npz"))
    if args.force_collect and data_dir.exists():
        print(f"[Phase 1] --force-collect: removing {data_dir}")
        shutil.rmtree(data_dir)
        train_dir_exists = False

    if args.skip_collect and train_dir_exists:
        print(f"[Phase 1] Skipping (data already at {data_dir})")
    else:
        cmd = [
            sys.executable, str(ROOT / "scripts" / "collect_data.py"),
            "--layout", str(layout_path),
            "--num-transitions", str(args.num_transitions),
            "--policy", args.policy,
            "--p-greedy", str(args.p_greedy),
            "--ghost-epsilon", str(args.ghost_epsilon),
            "--num-ghosts", str(args.num_ghosts),
            "--output-dir", str(data_dir),
            "--seed", str(args.collect_seed),
            "--val-fraction", str(args.val_fraction),
            "--min-spawn-dist", str(args.min_spawn_dist),
        ]
        if args.randomize_spawn:
            cmd.append("--randomize-spawn")
        print("[Phase 1] Collecting data:")
        print("  " + " ".join(cmd))
        subprocess.check_call(cmd)

    # -------- Phase 2: train world model --------
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "scripts" / "train_world_model.py"),
        "--data-dir", str(data_dir),
        "--config", str(ROOT / args.config),
        "--checkpoint-dir", str(ckpt_dir),
        "--device", args.device,
    ]
    if args.max_train_steps is not None:
        cmd += ["--max-train-steps", str(args.max_train_steps)]
    if args.eval_every is not None:
        cmd += ["--eval-every", str(args.eval_every)]
    if args.wandb:
        cmd.append("--wandb")
    print("[Phase 2] Training world model:")
    print("  " + " ".join(cmd))
    subprocess.check_call(cmd)

    print(f"\nDone. Checkpoints in: {ckpt_dir}")


if __name__ == "__main__":
    main()
