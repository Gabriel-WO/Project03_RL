"""
render.py - Pygame drawing for the Snake environment.

AI assistance disclosure: written with help from Claude (Anthropic); see README.md.

Kept separate from snake_env.py so training can run with no window at all
(much faster). Controls while a window is open:
    UP / DOWN arrows : speed up / slow down
    SPACE            : pause
    ESC or close     : quit
"""
import pygame

BG = (18, 18, 24)
GRID = (30, 30, 40)
HEAD = (80, 220, 120)
BODY = (40, 160, 90)
FOOD = (240, 80, 80)
TEXT = (230, 230, 235)
ACTION_NAMES = ["straight", "right", "left"]


class Renderer:
    def __init__(self, grid, cell=36, fps=15, title="Snake RL"):
        pygame.init()
        self.grid, self.cell, self.fps = grid, cell, fps
        self.hud_h = 96
        self.screen = pygame.display.set_mode((grid * cell, grid * cell + self.hud_h))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 17)
        self.clock = pygame.time.Clock()
        self.paused = False

    def draw(self, env, hud_lines=(), q_values=None):
        """Draw one frame. Returns False if the user asked to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key == pygame.K_UP:
                    self.fps = min(240, self.fps + 5)
                if event.key == pygame.K_DOWN:
                    self.fps = max(2, self.fps - 5)
        while self.paused:  # simple pause loop that still handles events
            if not self._pump_while_paused():
                return False

        c = self.cell
        self.screen.fill(BG)
        for i in range(self.grid + 1):  # faint grid lines
            pygame.draw.line(self.screen, GRID, (i * c, 0), (i * c, self.grid * c))
            pygame.draw.line(self.screen, GRID, (0, i * c), (self.grid * c, i * c))
        fx, fy = env.food
        pygame.draw.rect(self.screen, FOOD, (fx * c + 4, fy * c + 4, c - 8, c - 8), border_radius=8)
        for i, (x, y) in enumerate(env.snake):
            color = HEAD if i == 0 else BODY
            pygame.draw.rect(self.screen, color, (x * c + 2, y * c + 2, c - 4, c - 4), border_radius=6)

        y0 = self.grid * c + 6
        lines = list(hud_lines)
        if q_values is not None:  # show the agent's "thoughts" for the presentation
            best = int(max(range(3), key=lambda i: q_values[i]))
            lines.append("Q: " + "  ".join(
                ("[%s %.1f]" if i == best else "%s %.1f") % (ACTION_NAMES[i], q_values[i])
                for i in range(3)))
        for i, line in enumerate(lines):
            self.screen.blit(self.font.render(line, True, TEXT), (8, y0 + i * 21))
        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def _pump_while_paused(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE:
                    self.paused = False
        self.clock.tick(30)
        return True

    def close(self):
        pygame.quit()