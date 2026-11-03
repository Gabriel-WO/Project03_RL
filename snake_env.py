"""
snake_env.py - The Snake game as a reinforcement-learning ENVIRONMENT.

AI assistance disclosure: this project was written with help from Claude
(Anthropic). See the "AI Use Disclosure" section of README.md.

The agent NEVER sees the rules of Snake. It only receives:
    * a small state description (what is around the head, where the food is)
    * a reward number after every move
From those two things alone it has to discover that walls/body are bad and
food is good. That is the "game starts without knowing its rules" requirement.

Agent-environment loop (the core RL idea):
        state --> agent picks action --> env.step(action) --> (next_state, reward, done)
"""
import random
from collections import deque
from dataclasses import dataclass

# Absolute directions (clockwise order so "+1" means turn right).
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIRS = {UP: (0, -1), RIGHT: (1, 0), DOWN: (0, 1), LEFT: (-1, 0)}

# Action space: three RELATIVE moves. Relative actions (instead of 4 absolute
# directions) make it impossible to pick an instant 180-degree reversal and
# make learning faster because the policy is the same for every heading.
STRAIGHT, TURN_RIGHT, TURN_LEFT = 0, 1, 2
N_ACTIONS = 3
_TURN = {STRAIGHT: 0, TURN_RIGHT: 1, TURN_LEFT: -1}


@dataclass
class RewardConfig:
    """Reward function parameters. Tuning these is a big part of the project."""
    eat: float = 10.0      # ate food
    death: float = -10.0   # hit wall / own body / starved
    step: float = 0.0      # small cost per move (discourages dithering)
    toward: float = 0.0    # bonus when a move gets closer to the food
    away: float = 0.0      # (negative) penalty when a move gets farther away


# Two reward designs that we compare in experiments.py
PROFILES = {
    "sparse": RewardConfig(),  # only the big events matter
    "shaped": RewardConfig(step=-0.01, toward=0.1, away=-0.15),  # dense hints
}

# Two state designs that we compare in experiments.py
STATE_MODES = ("basic", "trap")


class SnakeEnv:
    def __init__(self, grid=12, reward=None, state_mode="basic", seed=None,
                 starvation_factor=60):
        assert state_mode in STATE_MODES
        self.grid = grid
        self.reward_cfg = reward or PROFILES["sparse"]
        self.state_mode = state_mode
        self.rng = random.Random(seed)
        # Episode ends if the snake wanders for too long without eating.
        self.starvation_factor = starvation_factor
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self):
        mid = self.grid // 2
        self.dir = RIGHT
        self.snake = deque([(mid, mid), (mid - 1, mid), (mid - 2, mid)])  # head first
        self.body = set(self.snake)
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.food = None
        self._place_food()
        return self.get_state()

    def step(self, action):
        """Apply one action. Returns (next_state, reward, done, info)."""
        cfg = self.reward_cfg
        self.dir = (self.dir + _TURN[action]) % 4
        head = self.snake[0]
        new = self._move(head, self.dir)
        self.steps += 1
        self.steps_since_food += 1

        eating = new == self.food
        tail = self.snake[-1]
        hit_wall = not self._in_bounds(new)
        # The tail cell is free to enter this turn (it moves away) unless we eat.
        hit_body = new in self.body and not (new == tail and not eating)
        if hit_wall or hit_body:
            return self.get_state(), cfg.death, True, {"reason": "wall" if hit_wall else "body"}

        old_dist = self._dist(head, self.food)

        # Move the snake.
        if not eating:
            self.body.discard(self.snake.pop())
        self.snake.appendleft(new)
        self.body.add(new)

        if eating:
            self.score += 1
            self.steps_since_food = 0
            reward = cfg.eat
            if len(self.snake) == self.grid * self.grid:  # board full = perfect game
                return self.get_state(), reward, True, {"reason": "win"}
            self._place_food()
        else:
            reward = cfg.step
            new_dist = self._dist(new, self.food)
            reward += cfg.toward if new_dist < old_dist else cfg.away

        if self.steps_since_food > self.starvation_factor * len(self.snake):
            return self.get_state(), cfg.death, True, {"reason": "starved"}
        return self.get_state(), reward, False, {}

    # ---------------------------------------------------------------- state
    def get_state(self):
        """
        Compress the board into a small integer (a bit-field) so the Q-table
        stays tiny and learning is fast.

        basic (11 bits): danger straight/right/left, heading one-hot (4),
                         food is up/right/down/left of head (4)
        trap  (14 bits): basic + 3 'trap' bits. A trap bit is 1 when moving
                         that way is NOT an immediate crash but leads into a
                         pocket with less free room than the snake's length
                         (found with a flood fill). This is our extension
                         beyond the standard tutorial state.
        """
        d, head = self.dir, self.snake[0]
        moves = [self._move(head, d), self._move(head, (d + 1) % 4), self._move(head, (d + 3) % 4)]
        feats = [self._blocked(p) for p in moves]
        feats += [d == UP, d == RIGHT, d == DOWN, d == LEFT]
        fx, fy = self.food
        hx, hy = head
        feats += [fy < hy, fx > hx, fy > hy, fx < hx]
        if self.state_mode == "trap":
            feats += [(not self._blocked(p)) and self._is_trap(p) for p in moves]
        state = 0
        for bit in feats:
            state = (state << 1) | int(bit)
        return state

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _move(pos, d):
        dx, dy = DIRS[d]
        return pos[0] + dx, pos[1] + dy

    def _in_bounds(self, p):
        return 0 <= p[0] < self.grid and 0 <= p[1] < self.grid

    def _blocked(self, p):
        """Wall or body. The tail is treated as free because it will move."""
        return (not self._in_bounds(p)) or (p in self.body and p != self.snake[-1])

    @staticmethod
    def _dist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _is_trap(self, start):
        """Flood fill from `start`; True if reachable room < snake length."""
        need = len(self.snake)
        seen, queue = {start}, deque([start])
        while queue and len(seen) < need:
            x, y = queue.popleft()
            for dx, dy in DIRS.values():
                n = (x + dx, y + dy)
                if n not in seen and not self._blocked(n):
                    seen.add(n)
                    queue.append(n)
        return len(seen) < need

    def _place_food(self):
        free = [(x, y) for x in range(self.grid) for y in range(self.grid)
                if (x, y) not in self.body]
        self.food = self.rng.choice(free)