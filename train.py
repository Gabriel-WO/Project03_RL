"""
train.py - Train the Q-learning agent on Snake.

AI assistance disclosure: written with help from Claude (Anthropic); see README.md.

Command-line examples
---------------------
    python train.py                                   # headless, fast
    python train.py --render-every 500                # watch every 500th episode
    python train.py --state trap --reward shaped      # try other designs

While watching a training run in the window:
    UP / DOWN : speed up / slow down       SPACE : pause
    RIGHT     : skip ahead (train a chunk of episodes without drawing)
    ENTER     : finish the rest of training at full speed
    ESC       : cancel training (the saved model is NOT overwritten)
"""
import argparse
import csv
import os
import time

import numpy as np

from agent import QAgent
from snake_env import PROFILES, STATE_MODES, SnakeEnv

HINTS = ["UP/DOWN speed   SPACE pause   RIGHT skip ahead",
         "ENTER finish training   ESC cancel"]


def evaluate(agent, env, episodes=100):
    """Run the greedy policy (no exploration, no learning). Returns scores."""
    scores = []
    for _ in range(episodes):
        s, done = env.reset(), False
        while not done:
            s, _, done, _ = env.step(agent.act(s, greedy=True))
        scores.append(env.score)
    return scores


def train(episodes=20000, state_mode="basic", reward="sparse", seed=0, grid=12,
          alpha=0.1, gamma=0.95, eps_decay=0.9997, render_every=0, fps=30,
          log_path=None, model_path=None, verbose=True, ui=False):
    """
    Train a Q-learning agent.
      render_every > 0 : show every Nth episode in a Pygame window (watch it learn)
      ui=True          : open a window even when not showing games, with a progress screen
    The model is only saved if training ran to completion (not cancelled).
    """
    env = SnakeEnv(grid=grid, reward=PROFILES[reward], state_mode=state_mode, seed=seed)
    agent = QAgent(alpha=alpha, gamma=gamma, eps_decay=eps_decay, seed=seed)

    renderer = None
    if render_every or ui:
        from render import Renderer  # only import pygame window code if needed
        renderer = Renderer(grid, fps=fps, title="Snake RL - training")

    skip_chunk = max(500, episodes // 10)           # how far RIGHT jumps ahead
    next_show = 1 if render_every else float("inf")  # next episode to draw
    finish, last_ui = False, 0.0
    history, record, running = [], 0, True

    for ep in range(1, episodes + 1):
        s, done, total = env.reset(), False, 0.0
        show = renderer is not None and not finish and ep >= next_show
        while not done and running:
            a = agent.act(s)                        # 1. choose action (epsilon-greedy)
            s2, r, done, _ = env.step(a)            # 2. environment responds
            agent.learn(s, a, r, s2, done)          # 3. learn from the experience
            total += r
            # stop drawing mid-episode once the user asks to skip / finish
            if show and not (renderer.skip_request or renderer.finish_request):
                hud = [f"Training: episode {ep:,}/{episodes:,}   speed {renderer.fps}"] + HINTS
                running = renderer.draw(env, hud, agent.q[s2])
            s = s2
        if not running:
            break
        agent.end_episode()
        record = max(record, env.score)
        history.append((ep, env.score, total, env.steps, agent.eps))
        if verbose and ep % 1000 == 0:
            recent = np.mean([h[1] for h in history[-1000:]])
            print(f"ep {ep:5d} | avg score (last 1000) {recent:5.2f} | "
                  f"record {record:3d} | eps {agent.eps:.3f} | states {len(agent.q)}")

        if renderer is not None:
            if renderer.finish_request:
                finish, renderer.finish_request = True, False
            if renderer.skip_request:
                renderer.skip_request = False
                next_show = (ep if show else next_show) + skip_chunk
            elif show:
                next_show = ep + render_every
            # Not drawing a game this episode: keep the window alive with a progress screen.
            if not show and time.time() - last_ui > 0.05:
                last_ui = time.time()
                recent = np.mean([h[1] for h in history[-500:]])
                lines = [f"avg score (last 500): {recent:.1f}      best: {record}",
                         f"exploration (epsilon): {agent.eps:.3f}      states learned: {len(agent.q):,}"]
                running = renderer.draw_progress("Training...", ep, episodes, lines, HINTS)
                if not running:
                    break

    if renderer:
        renderer.close()
    aborted = len(history) < episodes
    if log_path and not aborted:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "score", "total_reward", "steps", "epsilon"])
            w.writerows(history)
    if model_path and not aborted:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        agent.save(model_path)
    return agent, history


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--state", choices=STATE_MODES, default="safe")
    p.add_argument("--reward", choices=list(PROFILES), default="sparse")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid", type=int, default=12)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--eps-decay", type=float, default=0.9997)
    p.add_argument("--render-every", type=int, default=0, help="show every Nth episode (0 = headless)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--model", default=None, help="where to save the Q-table (default models/<state>_<reward>.pkl)")
    a = p.parse_args()

    tag = f"{a.state}_{a.reward}"
    model = a.model or f"models/{tag}.pkl"
    agent, hist = train(a.episodes, a.state, a.reward, a.seed, a.grid, a.alpha, a.gamma,
                        a.eps_decay, a.render_every, a.fps, f"results/log_{tag}.csv", model)
    if len(hist) < a.episodes:
        print("\nTraining cancelled; model was not saved.")
    else:
        env = SnakeEnv(grid=a.grid, reward=PROFILES[a.reward], state_mode=a.state, seed=999)
        scores = evaluate(agent, env, 100)
        print(f"\nGreedy evaluation over 100 games: mean {np.mean(scores):.2f}, max {max(scores)}")
        print(f"Saved model to {model}")