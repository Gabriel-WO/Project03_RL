"""
menu.py - Start menu (Pygame) so everything can be chosen in the window.

AI assistance disclosure: written with help from Claude (Anthropic); see README.md.

Choices:
    Agent design   : which state representation the agent uses
    Model          : use the saved model, or train a new one
    Show training  : watch the snake learn, or just show a progress bar
    Episodes       : how long to train

Controls: UP/DOWN select a row, LEFT/RIGHT change it (or click a row),
          ENTER or the Start button begins, ESC quits.
"""
import pygame

from render import DARK, FOOTER, FOOTER_TEXT, HEADER, LIGHT, SNAKE, SNAKE_DARK, WHITE

PANEL = (240, 248, 225)
INK = (40, 70, 25)
DIM = (140, 160, 125)

STATES = ["basic", "trap", "safe"]
STATE_INFO = {
    "basic": "11 bits: dangers, heading, food direction",
    "trap": "+ is the move leading into a too-small pocket?",
    "safe": "+ can the head still reach its tail? (best)",
}
EPISODE_CHOICES = [5000, 10000, 20000, 40000]


def run_menu(size, has_model):
    """
    Show the menu. `has_model(state)` says whether a saved model exists for that state.
    Returns {"state", "train", "watch", "episodes"} or None if the user quit.
    """
    pygame.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Snake RL")
    w, h = size
    title = pygame.font.SysFont("arial,helvetica,dejavusans", 30, bold=True)
    label = pygame.font.SysFont("arial,helvetica,dejavusans", 20, bold=True)
    small = pygame.font.SysFont("arial,helvetica,dejavusans", 14)
    clock = pygame.time.Clock()

    state_i, retrain, watch, ep_i, sel = STATES.index("safe"), False, True, 2, 0
    head_h = 64
    row_rects = [pygame.Rect(30, head_h + 28 + i * 78, w - 60, 66) for i in range(4)]
    start_rect = pygame.Rect(30, row_rects[-1].bottom + 18, w - 60, 56)

    def status():
        """Derived settings: is a saved model present, and will we train?"""
        saved = has_model(STATES[state_i])
        return saved, (retrain or not saved)

    def change(row, d):
        nonlocal state_i, retrain, watch, ep_i
        saved, training = status()
        if row == 0:
            state_i = (state_i + d) % len(STATES)
        elif row == 1 and saved:
            retrain = not retrain
        elif row == 2 and training:
            watch = not watch
        elif row == 3 and training:
            ep_i = (ep_i + d) % len(EPISODE_CHOICES)

    def result():
        saved, training = status()
        return {"state": STATES[state_i], "train": training, "watch": watch,
                "episodes": EPISODE_CHOICES[ep_i]}

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return None
                if event.key == pygame.K_RETURN:
                    return result()
                if event.key == pygame.K_UP:
                    sel = (sel - 1) % 4
                if event.key == pygame.K_DOWN:
                    sel = (sel + 1) % 4
                if event.key == pygame.K_LEFT:
                    change(sel, -1)
                if event.key == pygame.K_RIGHT:
                    change(sel, 1)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if start_rect.collidepoint(event.pos):
                    return result()
                for i, r in enumerate(row_rects):
                    if r.collidepoint(event.pos):
                        sel = i
                        change(i, -1 if event.pos[0] < r.centerx else 1)

        saved, training = status()
        # ---- draw
        screen.fill(HEADER)
        pygame.draw.rect(screen, HEADER, (0, 0, w, head_h))
        screen.blit(title.render("Snake RL", True, WHITE), (24, 14))
        body_h = h - head_h - 84
        for x in range(0, w, 40):  # checkerboard backdrop, like the game board
            for y in range(0, body_h, 40):
                c = LIGHT if (x // 40 + y // 40) % 2 == 0 else DARK
                pygame.draw.rect(screen, c, (x, head_h + y, 40, 40))

        model_value = ("Use saved" if not retrain else "Train new") if saved else "Train new"
        model_sub = "a saved model exists" if saved else "no saved model for this design yet"
        rows = [
            ("Agent design", STATES[state_i], STATE_INFO[STATES[state_i]], True),
            ("Model", model_value, model_sub, saved),
            ("Show training", "Yes" if watch else "No",
             "watch the snake learn live" if watch else "progress bar only (faster)", training),
            ("Training episodes", f"{EPISODE_CHOICES[ep_i]:,}", "more episodes = stronger agent", training),
        ]
        for i, (name, value, sub, enabled) in enumerate(rows):
            r = row_rects[i]
            pygame.draw.rect(screen, PANEL, r, border_radius=12)
            if i == sel:
                pygame.draw.rect(screen, SNAKE, r, width=3, border_radius=12)
            ink = INK if enabled else DIM
            screen.blit(label.render(name, True, ink), (r.x + 16, r.y + 10))
            screen.blit(small.render(sub, True, DIM if not enabled else (95, 125, 70)), (r.x + 16, r.y + 38))
            val = label.render(f"<  {value}  >" if enabled else value, True, ink)
            screen.blit(val, val.get_rect(midright=(r.right - 16, r.centery)))

        pygame.draw.rect(screen, SNAKE, start_rect, border_radius=14)
        txt = title.render("Start", True, WHITE)
        screen.blit(txt, txt.get_rect(center=start_rect.center))

        foot_y = h - 84
        pygame.draw.rect(screen, FOOTER, (0, foot_y, w, 84))
        for i, line in enumerate(["UP/DOWN select   LEFT/RIGHT change   (or click a row)",
                                  "ENTER start   ESC quit"]):
            screen.blit(small.render(line, True, FOOTER_TEXT), (10, foot_y + 10 + i * 20))
        pygame.display.flip()
        clock.tick(30)