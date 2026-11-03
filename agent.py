"""
agent.py - Tabular Q-learning agent.

AI assistance disclosure: written with help from Claude (Anthropic); see README.md.

Q-learning keeps a table  Q[state][action]  = "how much total future reward do
I expect if I take `action` in `state` and then act well afterwards?"

Update rule after every step (Bellman equation / temporal-difference learning):

    Q[s][a] <- Q[s][a] + alpha * ( r + gamma * max_a' Q[s'][a']  -  Q[s][a] )
                                   |_________ target _________|

    alpha  = learning rate     (how far we move toward the new estimate)
    gamma  = discount factor   (how much we care about future vs. immediate reward)

Action policy: epsilon-greedy. With probability epsilon explore (random move),
otherwise exploit (best known move). Epsilon decays so the agent explores a lot
early, then settles into its learned strategy.
"""
import pickle
import random
from collections import defaultdict

import numpy as np

from snake_env import N_ACTIONS


class QAgent:
    def __init__(self, alpha=0.1, gamma=0.9, eps_start=1.0, eps_min=0.01,
                 eps_decay=0.998, seed=None):
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_min, self.eps_decay = eps_start, eps_min, eps_decay
        self.q = defaultdict(lambda: np.zeros(N_ACTIONS))  # unseen state -> all zeros
        self.rng = random.Random(seed)

    def act(self, state, greedy=False):
        """Epsilon-greedy action selection (greedy=True turns exploration off)."""
        if not greedy and self.rng.random() < self.eps:
            return self.rng.randrange(N_ACTIONS)
        values = self.q[state]
        best = np.flatnonzero(values == values.max())  # break ties randomly
        return int(self.rng.choice(list(best)))

    def learn(self, s, a, r, s2, done):
        """One Q-learning (TD) update."""
        target = r if done else r + self.gamma * self.q[s2].max()
        self.q[s][a] += self.alpha * (target - self.q[s][a])

    def end_episode(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    # ---------------------------------------------------------- persistence
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"q": dict(self.q), "eps": self.eps}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q = defaultdict(lambda: np.zeros(N_ACTIONS), data["q"])
        self.eps = data.get("eps", self.eps_min)