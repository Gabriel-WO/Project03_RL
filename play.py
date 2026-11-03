"""
play.py - Watch a trained agent play Snake (no more learning, no exploration).

AI assistance disclosure: written with help from Claude (Anthropic); see README.md.

    python play.py                               # uses models/basic_sparse.pkl
    python play.py --model models/trap_sparse.pkl --state trap
"""
import argparse

from agent import QAgent
from snake_env import PROFILES, STATE_MODES, SnakeEnv


def play(model, state_mode="basic", reward="sparse", grid=12, fps=15, games=20):
    """Open a window and let the trained agent play `games` games. Returns best score."""
    from render import Renderer  # imported here so importing play.py never opens a window
    env = SnakeEnv(grid=grid, reward=PROFILES[reward], state_mode=state_mode)
    agent = QAgent()
    agent.load(model)
    renderer = Renderer(grid, fps=fps, title="Snake RL - trained agent")

    best, running = 0, True
    for game in range(1, games + 1):
        s, done = env.reset(), False
        while not done and running:
            action = agent.act(s, greedy=True)
            hud = [f"game {game}/{games}", f"score {env.score}   best {max(best, env.score)}",
                   "UP/DOWN speed   SPACE pause   ESC quit"]
            running = renderer.draw(env, hud, agent.q[s])
            s, _, done, _ = env.step(action)
        best = max(best, env.score)
        if not running:
            break
    renderer.close()
    return best


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/basic_sparse.pkl")
    p.add_argument("--state", choices=STATE_MODES, default="basic")
    p.add_argument("--reward", choices=list(PROFILES), default="sparse")
    p.add_argument("--grid", type=int, default=12)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--games", type=int, default=20)
    a = p.parse_args()
    print(f"Best score: {play(a.model, a.state, a.reward, a.grid, a.fps, a.games)}")