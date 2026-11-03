"""
train.py - Train the Q-learning agent on Snake.

AI assistance disclosure: written with help from Claude (Anthropic); see README.md.

Examples
--------
    python train.py                                   # headless, fast
    python train.py --render-every 100                # watch every 100th episode
    python train.py --state trap --reward shaped      # try other designs
    python train.py --episodes 5000 --seed 1
"""
import argparse
import csv
import os

import numpy as np

from agent import QAgent
from snake_env import PROFILES, STATE_MODES, SnakeEnv


def evaluate(agent, env, episodes=100):
    """Run the greedy policy (no exploration, no learning). Returns scores."""
    scores = []
    for _ in range(episodes):
        s, done = env.reset(), False
        while not done:
            s, _, done, _ = env.step(agent.act(s, greedy=True))
        scores.append(env.score)
    return scores


def train(episodes=3000, state_mode="basic", reward="sparse", seed=0, grid=12,
          alpha=0.1, gamma=0.9, eps_decay=0.998, render_every=0, fps=30,
          log_path=None, model_path=None, verbose=True):
    env = SnakeEnv(grid=grid, reward=PROFILES[reward], state_mode=state_mode, seed=seed)
    agent = QAgent(alpha=alpha, gamma=gamma, eps_decay=eps_decay, seed=seed)

    renderer = None
    if render_every:
        from render import Renderer  # only import pygame window code if needed
        renderer = Renderer(grid, fps=fps, title="Snake RL - training")

    history, record, running = [], 0, True
    for ep in range(1, episodes + 1):
        s, done, total = env.reset(), False, 0.0
        show = renderer is not None and (ep % render_every == 0 or ep == 1)
        while not done and running:
            a = agent.act(s)                        # 1. choose action (epsilon-greedy)
            s2, r, done, _ = env.step(a)            # 2. environment responds
            agent.learn(s, a, r, s2, done)          # 3. learn from the experience
            total += r
            if show:
                hud = [f"episode {ep}/{episodes}   epsilon {agent.eps:.3f}",
                       f"score {env.score}   record {max(record, env.score)}",
                       f"states learned: {len(agent.q)}"]
                running = renderer.draw(env, hud, agent.q[s2])
            s = s2
        if not running:
            break
        agent.end_episode()
        record = max(record, env.score)
        history.append((ep, env.score, total, env.steps, agent.eps))
        if verbose and ep % 250 == 0:
            recent = np.mean([h[1] for h in history[-250:]])
            print(f"ep {ep:5d} | avg score (last 250) {recent:5.2f} | "
                  f"record {record:3d} | eps {agent.eps:.3f} | states {len(agent.q)}")

    if renderer:
        renderer.close()
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "score", "total_reward", "steps", "epsilon"])
            w.writerows(history)
    if model_path:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        agent.save(model_path)
    return agent, history


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--state", choices=STATE_MODES, default="basic")
    p.add_argument("--reward", choices=list(PROFILES), default="sparse")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid", type=int, default=12)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--eps-decay", type=float, default=0.998)
    p.add_argument("--render-every", type=int, default=0, help="show every Nth episode (0 = headless)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--model", default=None, help="where to save the Q-table (default models/<state>_<reward>.pkl)")
    a = p.parse_args()

    tag = f"{a.state}_{a.reward}"
    model = a.model or f"models/{tag}.pkl"
    agent, hist = train(a.episodes, a.state, a.reward, a.seed, a.grid, a.alpha, a.gamma,
                        a.eps_decay, a.render_every, a.fps, f"results/log_{tag}.csv", model)
    env = SnakeEnv(grid=a.grid, reward=PROFILES[a.reward], state_mode=a.state, seed=999)
    scores = evaluate(agent, env, 100)
    print(f"\nGreedy evaluation over 100 games: mean {np.mean(scores):.2f}, max {max(scores)}")
    print(f"Saved model to {model}")