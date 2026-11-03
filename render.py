"""
render.py - Pygame drawing for the Snake environment, styled after Google's Snake game.

AI assistance disclosure: written with help from Claude (Anthropic); see README.md.

Kept separate from snake_env.py so training can run with no window at all
(much faster). Controls while a window is open:
    UP / DOWN arrows : speed up / slow down
    SPACE            : pause
    ESC or close     : quit
"""
import pygame

# Google-Snake-inspired palette
HEADER = (74, 117, 44)
LIGHT = (170, 215, 81)
DARK = (162, 209, 73)
SNAKE = (70, 116, 233)
SNAKE_DARK = (52, 92, 200)
APPLE = (231, 71, 29)
APPLE_HI = (255, 140, 110)
LEAF = (52, 130, 40)
WHITE = (255, 255, 255)
PUPIL = (20, 20, 30)
FOOTER = (60, 98, 36)
FOOTER_TEXT = (225, 240, 205)
ACTION_NAMES = ["straight", "right", "left"]
# (dx, dy) for UP, RIGHT, DOWN, LEFT - matches snake_env
_VEC = [(0, -1), (1, 0), (0, 1), (-1, 0)]


class Renderer:
    def __init__(self, grid, cell=40, fps=15, title="Snake RL"):
        pygame.init()
        self.grid, self.cell, self.fps = grid, cell, fps
        self.header_h, self.footer_h = 64, 46
        self.width = grid * cell
        self.screen = pygame.display.set_mode(
            (self.width, self.header_h + grid * cell + self.footer_h))
        pygame.display.set_caption(title)
        self.big = pygame.font.SysFont("arial,helvetica,dejavusans", 30, bold=True)
        self.small = pygame.font.SysFont("arial,helvetica,dejavusans", 15)
        self.clock = pygame.time.Clock()
        self.paused = False
        self.best = 0

    # ------------------------------------------------------------- drawing
    def draw(self, env, hud_lines=(), q_values=None):
        """Draw one frame. Returns False if the user asked to quit."""
        if not self._handle_events():
            return False
        self.best = max(self.best, env.score)
        self._draw_scene(env, hud_lines, q_values)
        if self.paused:
            self._draw_pause_overlay()
            pygame.display.flip()
            while self.paused:  # wait here, still responding to keys
                if not self._handle_events():
                    return False
                self.clock.tick(30)
            return True
        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def _draw_scene(self, env, hud_lines, q_values):
        c, oy = self.cell, self.header_h
        self.screen.fill(HEADER)
        self._draw_header(env.score)
        # checkerboard
        for x in range(self.grid):
            for y in range(self.grid):
                color = LIGHT if (x + y) % 2 == 0 else DARK
                pygame.draw.rect(self.screen, color, (x * c, oy + y * c, c, c))
        self._draw_apple(*env.food)
        self._draw_snake(env)
        self._draw_footer(hud_lines, q_values)

    def _draw_header(self, score):
        h = self.header_h
        # apple icon + current score
        self._apple_icon(30, h // 2, 14)
        self._text(self.big, str(score), (56, h // 2), WHITE, anchor="midleft")
        # trophy + best score
        self._trophy(self.width // 2 + 18, h // 2)
        self._text(self.big, str(self.best), (self.width // 2 + 44, h // 2), WHITE, anchor="midleft")

    def _draw_apple(self, gx, gy):
        c = self.cell
        cx, cy = int(gx * c + c / 2), int(self.header_h + gy * c + c / 2 + 1)
        r = int(c * 0.36)
        self._apple_icon(cx, cy, r)

    def _apple_icon(self, cx, cy, r):
        pygame.draw.circle(self.screen, APPLE, (cx, cy + 1), r)
        pygame.draw.circle(self.screen, APPLE_HI, (cx - r // 3, cy - r // 3), max(2, r // 4))
        pygame.draw.line(self.screen, (90, 60, 30), (cx, cy - r + 2), (cx + 2, cy - r - r // 3), 3)
        leaf = [(cx + 2, cy - r), (cx + r // 2 + 3, cy - r - r // 3), (cx + r // 3, cy - r + 3)]
        pygame.draw.polygon(self.screen, LEAF, leaf)

    def _trophy(self, cx, cy):
        col = (255, 213, 79)
        pygame.draw.rect(self.screen, col, (cx - 10, cy - 12, 20, 14), border_radius=4)
        pygame.draw.rect(self.screen, col, (cx - 3, cy + 2, 6, 7))
        pygame.draw.rect(self.screen, col, (cx - 9, cy + 8, 18, 4), border_radius=2)
        pygame.draw.circle(self.screen, col, (cx - 12, cy - 6), 5, 2)
        pygame.draw.circle(self.screen, col, (cx + 12, cy - 6), 5, 2)

    def _draw_snake(self, env):
        c, oy = self.cell, self.header_h
        body = list(env.snake)
        half = int(c * 0.40)  # half-thickness of the snake

        def center(p):
            return p[0] * c + c // 2, oy + p[1] * c + c // 2

        # joints between consecutive segments so the body looks continuous
        for a, b in zip(body, body[1:]):
            (ax, ay), (bx, by) = center(a), center(b)
            rect = pygame.Rect(min(ax, bx) - half, min(ay, by) - half,
                               abs(ax - bx) + 2 * half, abs(ay - by) + 2 * half)
            pygame.draw.rect(self.screen, SNAKE, rect)
        for i, p in enumerate(body):
            x, y = center(p)
            pygame.draw.rect(self.screen, SNAKE, (x - half, y - half, 2 * half, 2 * half),
                             border_radius=int(half * 0.8))
        self._draw_eyes(center(body[0]), env.dir, half)

    def _draw_eyes(self, center, direction, half):
        cx, cy = center
        fx, fy = _VEC[direction]
        px, py = -fy, fx  # perpendicular to heading
        er = max(5, int(self.cell * 0.17))
        for side in (-1, 1):
            ex = cx + fx * half * 0.10 + px * half * 0.52 * side
            ey = cy + fy * half * 0.10 + py * half * 0.52 * side
            pygame.draw.circle(self.screen, WHITE, (int(ex), int(ey)), er)
            pygame.draw.circle(self.screen, PUPIL, (int(ex + fx * er * 0.35), int(ey + fy * er * 0.35)),
                               max(2, er // 2))

    def _draw_footer(self, hud_lines, q_values):
        y0 = self.header_h + self.grid * self.cell
        pygame.draw.rect(self.screen, FOOTER, (0, y0, self.width, self.footer_h))
        lines = list(hud_lines)[:1]  # one info line (episode / game number)
        if q_values is not None:  # the agent's "thoughts", handy for the presentation
            best = int(max(range(3), key=lambda i: q_values[i]))
            lines.append("   ".join(
                (">%s %.1f" if i == best else "%s %.1f") % (ACTION_NAMES[i], q_values[i])
                for i in range(3)))
        for i, line in enumerate(lines):
            self._text(self.small, line, (10, y0 + 6 + i * 18), FOOTER_TEXT)

    def _draw_pause_overlay(self):
        shade = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 110))
        self.screen.blit(shade, (0, 0))
        self._text(self.big, "PAUSED", (self.width // 2, self.screen.get_height() // 2), WHITE,
                   anchor="center")

    def _text(self, font, text, pos, color, anchor="topleft"):
        surf = font.render(text, True, color)
        self.screen.blit(surf, surf.get_rect(**{anchor: pos}))

    # -------------------------------------------------------------- events
    def _handle_events(self):
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
        return True

    def close(self):
        pygame.quit()