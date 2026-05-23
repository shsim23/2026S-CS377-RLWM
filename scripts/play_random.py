"""Run a random policy and print aggregate statistics over N episodes."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pacman_env import PacmanEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="layouts/train/medium_classic.txt")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = PacmanEnv(layout_path=args.layout)

    deaths, wins, lengths, rewards = 0, 0, [], []

    for ep in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                if info["event"]["died"]:
                    deaths += 1
                if info["event"]["won"]:
                    wins += 1
                lengths.append(info["step"])
                rewards.append(ep_reward)
                break

    N = args.num_episodes
    print(f"Episodes: {N}")
    print(f"Death rate:    {deaths / N * 100:.1f}%")
    print(f"Win rate:      {wins / N * 100:.1f}%")
    print(f"Avg length:    {sum(lengths) / N:.1f}")
    print(f"Avg reward:    {sum(rewards) / N:.2f}")


if __name__ == "__main__":
    main()
