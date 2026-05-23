"""Human keyboard-controlled play. Requires pygame."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame
from pacman_env import PacmanEnv, Action

LAYOUT = "layouts/train/medium_classic.txt"
KEY_MAP = {
    pygame.K_UP:    Action.UP,
    pygame.K_DOWN:  Action.DOWN,
    pygame.K_LEFT:  Action.LEFT,
    pygame.K_RIGHT: Action.RIGHT,
}


def main():
    env = PacmanEnv(layout_path=LAYOUT, render_mode="human")
    obs, info = env.reset(seed=42)
    env.render()

    action = Action.NOOP
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                action = KEY_MAP.get(event.key, Action.NOOP)

        obs, reward, terminated, truncated, info = env.step(int(action))
        env.render()
        action = Action.NOOP

        if terminated or truncated:
            print(f"Episode ended — reward={info['score']:.1f}  died={info['event']['died']}  won={info['event']['won']}")
            obs, info = env.reset(seed=0)

    env.close()


if __name__ == "__main__":
    main()
