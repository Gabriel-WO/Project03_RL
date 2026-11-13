# Project03_RL: Snake that Learns Its Own Rules

A Pygame version of Snake where the agent is **never told the rules**. It only sees a small
description of its surroundings and a reward number after each move, and learns to play with
**tabular Q-learning**.

> **Author:** _(your name)_  |  **Class:** ADV CS III: Machine Learning and AI  |  **Teacher:** Mr. Cochran
> **Collaborators / help received:** _(list classmates who helped you, if any)_

## Quick start

1. Install Python 3.9 or newer, then install the libraries: `pip install -r requirements.txt`
   (`pygame` for the game window, `numpy` for the Q-table math, `matplotlib` for the experiment plots).
2. Open the project folder in PyCharm (or a terminal in that folder) and run **`main.py`**.
3. Use the start menu to choose what to do (next section). A trained model is already included
   in `models/`, so you can pick **Use saved** and watch the agent play right away.

Run it from inside the project folder, because the program looks for `models/` and writes to `results/` relative to where it runs.

## Using the program (`main.py` start menu)

Everything is chosen in the window, with the arrow keys or by clicking a row:

| Menu row | Choices | What it does |
|---|---|---|
| **Agent design** | `basic` / `trap` / `safe` | Which state representation the agent sees (see "The RL design"). `safe` is the strongest |
| **Model** | Use saved / Train new | Load the saved model for that design, or train a fresh one. If no saved model exists yet for the design, training is forced |
| **Show training** | Yes / No | Yes = watch the snake learn live. No = just a progress bar (faster) |
| **Training episodes** | 5,000 / 10,000 / 20,000 / 40,000 | How many games the agent trains for. More episodes = stronger agent, longer wait |

Press **Start** (or ENTER). If you chose to train, training runs and the model is saved automatically; then the
trained agent plays 20 games. When it finishes (or you press ESC), you return to the menu. Press ESC at the menu to quit.

### Controls

| Screen | Keys |
|---|---|
| Menu | UP/DOWN select a row, LEFT/RIGHT change it (or click a row), ENTER start, ESC quit |
| Watching training | UP/DOWN speed up / slow down, **RIGHT skip ahead** (trains a chunk of episodes without drawing), **ENTER finish training at full speed**, SPACE pause, ESC cancel |
| Training progress screen | Same keys as above (shown when "Show training" is No, and between watched games) |
| Agent playing | UP/DOWN speed, SPACE pause, ESC back to the menu |

Cancelling training (ESC or closing the window) **never overwrites** your saved model; a model is only saved when training finishes.

## Creating and using models

**What a model is.** A model is the agent's learned Q-table, saved as a pickle file named
`models/<state>_<reward>.pkl` (for example `models/safe_sparse.pkl`). The name comes from the agent design and the reward
function, so each combination keeps its own model. Training again with the same settings replaces that file.

**To create (train) a model**

- *In the window:* run `main.py`, set **Model = Train new**, pick the design and number of episodes, press Start.
- *From a terminal:* `python train.py --state safe --episodes 20000` (add `--render-every 500` to watch every 500th game).
- Training also writes `results/log_<state>_<reward>.csv` with the score, total reward, steps, and epsilon for every episode,
  which is handy for making your own graphs.
- Rough training time for 20,000 episodes without watching (it varies by computer): `basic` a few seconds, `trap` about 2 minutes, `safe` about 6 minutes.
  Watching training is slower. Use **RIGHT** to skip ahead or **ENTER** to finish at full speed.
- Shorter runs give weaker agents. For a quick demo, 5,000 episodes works, but expect a lower score than the 20,000-episode model.

**To use (play) a saved model**

- *In the window:* run `main.py`, set **Model = Use saved**, press Start.
- *From a terminal:* `python play.py --model models/safe_sparse.pkl --state safe`.
- The agent plays greedily: it always picks the move with the best learned value, with no exploration and no more learning.
  The footer of the window shows its Q-value for each move (the `>` marks the one it picks).

**Important:** a model only works with the same `--state` design it was trained with, because the state numbers mean different
things in each design. The menu handles this automatically; on the command line, make `--state` (and `--reward`) match the model.
If you change `GRID` in `main.py`, train a new model for that board size.

**Other settings you can change**

- `main.py` (top of file): `REWARD` (`"sparse"` or `"shaped"`), `GRID` (board size), `SEED`, `PLAY_GAMES`, `WATCH_EVERY`.
- `snake_env.py`: the `PROFILES` dictionary holds the reward values (eat, die, step, closer/farther from food).
- `train.py` / `agent.py`: learning rate `alpha`, discount `gamma`, and epsilon decay (also available as command-line flags).

## Command-line reference

| Command | Options |
|---|---|
| `python train.py` | `--episodes N` (default 20000), `--state basic/trap/safe` (default safe), `--reward sparse/shaped`, `--seed N`, `--grid N`, `--alpha X`, `--gamma X`, `--eps-decay X`, `--render-every N` (0 = no window), `--fps N`, `--model PATH` |
| `python play.py` | `--model PATH`, `--state ...`, `--reward ...`, `--grid N`, `--fps N`, `--games N` |
| `python experiments.py` | `--episodes N` (default 4000), `--seeds N` (default 3), `--window N` (smoothing for the plot). Trains every state/reward design and writes `results/experiment_results.csv` and `results/comparison.png`. It takes several minutes, and longer if you raise `--episodes` |

## Files

```
Project03_RL/
  main.py          start menu loop (run this)
  menu.py          the menu screen
  snake_env.py     the game + state + rewards (the environment)
  agent.py         the Q-learning agent
  train.py         training loop
  play.py          watch a trained agent
  render.py        all Pygame drawing
  experiments.py   compares designs, makes the plot
  requirements.txt libraries to install
  models/          saved Q-tables (.pkl)
  results/         training logs, experiment table, comparison plot
```

| File | What it does |
|---|---|
| `main.py` | The program you run. Shows the menu, then trains and/or plays based on your choices, and loops back to the menu. Holds a few fixed settings (`REWARD`, `GRID`, `SEED`, `PLAY_GAMES`, `WATCH_EVERY`) |
| `menu.py` | `run_menu()` draws the start menu and returns your choices (design, train or use saved, show training, episodes). Forces training when no saved model exists |
| `snake_env.py` | The **environment**. `SnakeEnv` runs the game rules (`reset()`, `step(action)`), builds the state number (`get_state()`, including the flood-fill checks), and computes rewards from `RewardConfig` / `PROFILES`. The agent never sees these rules, only the state and reward it returns |
| `agent.py` | The **agent**. `QAgent` stores the Q-table, picks moves with `act()` (epsilon-greedy), learns with `learn()` (the Bellman update), and `save()` / `load()` the model file |
| `train.py` | `train()` runs the agent-environment loop for many episodes, logs results to CSV, saves the model, and handles the live viewing controls (speed, skip ahead, finish, cancel). `evaluate()` measures a trained agent over greedy games. Also works as a command-line tool |
| `play.py` | `play()` loads a saved model and shows it playing in a window. Also works as a command-line tool |
| `render.py` | `Renderer` draws the Google-Snake-style board, header with score and best score, the training progress screen, and the footer with key hints and live Q-values. Kept separate so training can run with no window |
| `experiments.py` | Trains every state/reward design over several random seeds, evaluates each, and saves the results table and learning-curve plot |
| `requirements.txt` | The three libraries to install (`pygame`, `numpy`, `matplotlib`) |
| `models/` | Saved Q-tables (`basic_sparse.pkl`, `trap_sparse.pkl`, `safe_sparse.pkl`) |
| `results/` | `log_*.csv` training logs, `experiment_results.csv`, `comparison.png` |

## The RL design

**Agent-environment loop.** Each step: the environment gives a state, the agent picks an action
(epsilon-greedy), the environment returns `(next_state, reward, done)`, and the agent updates its Q-table.
The agent has no access to the game code. It discovers that walls are bad purely because crashing returns -10.

**Action space (3 actions, relative to heading):** go straight, turn right, turn left.
Relative actions mean the learned policy works for every heading and a 180-degree reversal is impossible.

**State space.** Each state is a bit-field packed into one integer, which keeps the Q-table tiny:

| Design | Bits | Contents |
|---|---|---|
| `basic` | 11 | danger straight/right/left, heading (4), food is up/right/down/left of the head (4) |
| `trap` (my extension) | 14 | `basic` + 3 "trap" bits |
| `safe` (my extension, best) | 17 | `trap` + 3 "cut-off" bits |

A **trap bit** is 1 when a move is *not* an immediate crash but leads into a region with less free
space than the snake's current length. I compute it with a flood fill (BFS). This lets the agent see
"that move is safe now but I'll box myself in," which the standard 11-bit state can't express.

A **cut-off bit** is 1 when a move is not an immediate crash but afterwards the head can no longer reach the
snake's own tail (another flood fill). That means the snake has sealed itself into a closed pocket, the classic
way long snakes die. Both extra features are computed by the environment, so the agent still learns *what to do* about them only from rewards.

**Reward function** (two versions compared):

| Event | `sparse` | `shaped` |
|---|---|---|
| Eat food | +10 | +10 |
| Die (wall, body, or starvation timeout) | -10 | -10 |
| Each step | 0 | -0.01 |
| Move closer to food / farther away | 0 / 0 | +0.1 / -0.15 |

**Learning algorithm.** Q-learning with the Bellman update
`Q[s][a] += alpha * (r + gamma * max Q[s'] - Q[s][a])`, with `alpha = 0.1` and `gamma = 0.95`.
**Policy:** epsilon-greedy, epsilon decaying from 1.0 to 0.01 (x0.9997 per episode, 20,000 episodes), so the agent
explores early and exploits later. A starvation timeout (60 x snake length steps without food) stops it from looping forever.

## Results

Board 12x12, 4000 training episodes, 5 random seeds each. After training, each agent plays 100 greedy games (no exploration).

| Design | Mean score (food eaten) | Std across seeds |
|---|---|---|
| basic state + sparse reward | 19.69 | 1.15 |
| basic state + shaped reward | 20.26 | 1.33 |
| **trap state + sparse reward** | **36.60** | 1.18 |
| trap state + shaped reward | 35.83 | 1.79 |

![learning curves](results/comparison.png)

**Round 2: richer state and longer training** (single seed, 20,000 episodes, 200 greedy games; the
earlier table used 4,000 episodes, so compare across rows with care):

| State | gamma | Mean score | Best game |
|---|---|---|---|
| trap, 4,000 episodes (above) | 0.9 | 36.6 | 53 |
| safe, 4,000 episodes | 0.9 | 32.8 | 53 |
| trap, 20,000 episodes | 0.95 | 41.3 | 62 |
| **safe, 20,000 episodes** | **0.95** | **46.9** | 78 |
| safe, 20,000 episodes | 0.99 | 37.9 | 64 |

The `safe` state has about 2,200 reachable states instead of about 250, so it learns *slower* at first (worse at 4,000
episodes) but ends up best once it has enough training. The shipped model (`models/safe_sparse.pkl`) averages 46.7 on 100 greedy games, best 71.
Raising gamma to 0.99 hurt, probably because rewards far in the future are too noisy to learn from.

**What I learned from the experiments**

1. **State design mattered most.** Adding three flood-fill bits nearly doubled the score (about 20 to about 37). The basic agent mostly dies by trapping itself once the snake gets long, because it cannot see that danger.
2. **Reward shaping barely helped.** The shaped and sparse rewards are within the seed-to-seed noise of each other. Distance hints did not matter because the state already tells the agent which way the food is. I expected shaping to help more, so this was a useful surprise.
3. **Tuning history:** I started with the standard 11-bit state and a sparse reward (mean about 20), then tried reward shaping (no real gain), then changed the state representation (big gain). The lesson is to diagnose *why* the agent dies before tuning numbers.

**Limitations / future work:** this agent is good but **not optimal**. A perfect Snake game fills all 144 cells of the 12x12
board; this agent averages about 47 and its best game is about 70-78. The state only describes the immediate neighborhood plus two
flood-fill checks, so it cannot plan a long route through a crowded board, and many different board positions share one Q-table entry.
That is the price of a small, inspectable table. Ideas: add a Q-table for longer lookahead, try Deep Q-Learning (DQN) for comparison, or give the snake a bigger board.

## What makes this different from a standard tutorial

- Tabular Q-learning with a compact bit-field state instead of a neural network, so every learned value can be inspected (the HUD shows the live Q-values).
- A custom **flood-fill trap feature** in the state, plus a controlled experiment (4 designs x 5 seeds) showing its effect.
- Relative action space, starvation timeout, and a headless/visual split so training is fast but can be watched live.
- A second flood-fill feature (the **cut-off** bit) and a start menu where you can pick the agent design, retrain or reuse a model, and watch, speed up, or skip through training.

## AI Use Disclosure (Honor Code)

> **Edit this section so it is accurate and in your own words before submitting.**

I used **Claude (Anthropic)** to help build this project. Claude helped me plan the project structure and wrote the initial
versions of the code in `snake_env.py`, `agent.py`, `train.py`, `play.py`, `render.py`, `menu.py`, `main.py`, and `experiments.py`, along with the first
draft of this README. It also ran the experiments that produced the results above. Each source file begins with a comment noting this.

_(Add how YOU used it and what you changed or learned: for example, which parameters you tuned, what you tested yourself,
and which parts you can explain from memory.)_

Other resources consulted: _(list any websites from the project handout you actually used)_
