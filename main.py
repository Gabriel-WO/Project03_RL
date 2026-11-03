"""
main.py - Run this file (press Run in PyCharm) to train the Snake agent and watch it play.
"""
import os

import numpy as np

from play import play
from snake_env import PROFILES, SnakeEnv
from train import evaluate, train

# ----------------------------- SETTINGS (edit these) -----------------------------
STATE_MODE = "trap"        # "basic" (11-bit state) or "trap" (adds flood-fill trap bits)
REWARD = "sparse"          # "sparse" or "shaped"
EPISODES = 4000            # number of training games
RETRAIN = False            # True = always retrain, even if a saved model exists
WATCH_TRAINING = True      # True = show a window while training (slower, good for demos)
WATCH_EVERY = 100          # when watching, display every Nth training episode
PLAY_GAMES = 20            # how many games the trained agent plays at the end
GRID = 12                  # board size
SEED = 0
# ----------------------------------------------------------------------------------

MODEL_PATH = f"models/{STATE_MODE}_{REWARD}.pkl"


def main():
    if RETRAIN or not os.path.exists(MODEL_PATH):
        print(f"Training ({STATE_MODE} state, {REWARD} reward, {EPISODES} episodes)...")
        agent, _ = train(EPISODES, STATE_MODE, REWARD, seed=SEED, grid=GRID,
                         render_every=WATCH_EVERY if WATCH_TRAINING else 0,
                         log_path=f"results/log_{STATE_MODE}_{REWARD}.csv",
                         model_path=MODEL_PATH)
        env = SnakeEnv(grid=GRID, reward=PROFILES[REWARD], state_mode=STATE_MODE, seed=999)
        scores = evaluate(agent, env, 100)
        print(f"Greedy evaluation over 100 games: mean {np.mean(scores):.2f}, max {max(scores)}")
    else:
        print(f"Found saved model {MODEL_PATH} (set RETRAIN = True to train again).")

    print("Playing with the trained agent. UP/DOWN = speed, SPACE = pause, ESC = quit.")
    best = play(MODEL_PATH, STATE_MODE, REWARD, grid=GRID, games=PLAY_GAMES)
    print(f"Best score: {best}")


if __name__ == "__main__":
    main()