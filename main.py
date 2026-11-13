"""
main.py - Run this file (press Run in PyCharm). Everything is chosen in the window.

AI assistance disclosure: written with help from Claude (Anthropic); see README.md.

Flow (loops until you quit from the menu):
    1. menu.py       : choose agent design, use saved model vs. train new,
                       show training or not, number of episodes
    2. train.py      : trains (if chosen). While watching you can speed up, skip ahead, or finish
    3. play.py       : the trained agent plays; ESC returns to the menu

The work is split across the other files:
    snake_env.py  - game rules, state, rewards     agent.py - Q-learning agent
    render.py     - Pygame drawing                 menu.py  - start menu
"""
import os

import numpy as np

from menu import run_menu
from play import play
from render import Renderer
from snake_env import PROFILES, SnakeEnv
from train import evaluate, train

# ------------------ fixed settings (the menu covers the rest) ------------------
REWARD = "sparse"    # "sparse" or "shaped" (see snake_env.PROFILES)
GRID = 12            # board size
SEED = 0
PLAY_GAMES = 20      # games the trained agent plays before returning to the menu
WATCH_EVERY = 250    # when watching training, draw every Nth episode
# -------------------------------------------------------------------------------


def model_path(state):
    return f"models/{state}_{REWARD}.pkl"


def main():
    size = Renderer.window_size(GRID)
    while True:
        choice = run_menu(size, lambda state: os.path.exists(model_path(state)))
        if choice is None:  # user quit from the menu
            break
        state, path = choice["state"], model_path(choice["state"])

        if choice["train"]:
            n = choice["episodes"]
            print(f"Training ({state} state, {REWARD} reward, {n:,} episodes)...")
            agent, history = train(
                n, state, REWARD, seed=SEED, grid=GRID,
                render_every=WATCH_EVERY if choice["watch"] else 0, ui=True,
                log_path=f"results/log_{state}_{REWARD}.csv", model_path=path)
            if len(history) < n:
                print("Training cancelled - the saved model was left unchanged.")
                continue
            env = SnakeEnv(grid=GRID, reward=PROFILES[REWARD], state_mode=state, seed=999)
            scores = evaluate(agent, env, 100)
            print(f"Greedy evaluation over 100 games: mean {np.mean(scores):.2f}, max {max(scores)}")

        best = play(path, state, REWARD, grid=GRID, games=PLAY_GAMES)
        print(f"Best score this session: {best}")


if __name__ == "__main__":
    main()