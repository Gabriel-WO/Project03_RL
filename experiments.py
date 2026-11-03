"""
experiments.py - Compare state and reward designs (the "iteration and tuning"
evidence for the rubric).

AI assistance disclosure: written with help from Claude (Anthropic); see README.md.

We train 4 designs (2 state representations x 2 reward functions) over several
random seeds, then evaluate each trained agent greedily and plot learning curves.

    python experiments.py                       # default: 3 seeds, 4000 episodes
    python experiments.py --seeds 5 --episodes 6000
"""
import argparse
import csv
import itertools
import os

import matplotlib
matplotlib.use("Agg")  # no window needed
import matplotlib.pyplot as plt
import numpy as np

from snake_env import PROFILES, STATE_MODES, SnakeEnv
from train import evaluate, train

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, default=4000)
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--window", type=int, default=100, help="moving-average window for curves")
a = p.parse_args()

os.makedirs("results", exist_ok=True)
configs = list(itertools.product(STATE_MODES, PROFILES))  # (state, reward) pairs
curves, finals = {}, {}

for state_mode, reward in configs:
    name = f"{state_mode}+{reward}"
    curves[name], finals[name] = [], []
    for seed in range(a.seeds):
        agent, hist = train(a.episodes, state_mode, reward, seed=seed, verbose=False)
        curves[name].append([h[1] for h in hist])
        env = SnakeEnv(reward=PROFILES[reward], state_mode=state_mode, seed=1000 + seed)
        finals[name].append(np.mean(evaluate(agent, env, 100)))
        print(f"{name:14s} seed {seed}: greedy mean score {finals[name][-1]:.2f}")

# ---- results table
with open("results/experiment_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["config", "mean_greedy_score", "std_across_seeds", "per_seed"])
    for name, vals in finals.items():
        w.writerow([name, f"{np.mean(vals):.2f}", f"{np.std(vals):.2f}",
                    " ".join(f"{v:.2f}" for v in vals)])
print("\nconfig          mean   std")
for name, vals in finals.items():
    print(f"{name:14s} {np.mean(vals):6.2f} {np.std(vals):5.2f}")


# ---- learning-curve plot (mean of seeds, smoothed)
def smooth(x, k):
    return np.convolve(x, np.ones(k) / k, mode="valid")


fig, ax = plt.subplots(figsize=(8, 5))
for name, runs in curves.items():
    m = smooth(np.mean(runs, axis=0), a.window)
    ax.plot(np.arange(len(m)) + a.window, m, label=name, linewidth=2)
ax.set_xlabel("training episode")
ax.set_ylabel(f"score (food eaten), {a.window}-episode average")
ax.set_title("Snake Q-learning: state and reward design comparison")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("results/comparison.png", dpi=150)
print("Saved results/comparison.png and results/experiment_results.csv")