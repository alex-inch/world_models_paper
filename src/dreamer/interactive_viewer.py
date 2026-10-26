"""Pygame controls and rendering for the Dreamer inspector."""

from dataclasses import dataclass, field

import numpy as np
import pygame
from scipy.special import softmax

BACKGROUND = (18, 22, 28)
PANEL = (27, 33, 42)
TEXT = (232, 237, 244)
MUTED = (162, 176, 194)
LINE = (66, 78, 96)
BLUE = (112, 184, 249)
ORANGE = (255, 182, 111)
GREEN = (131, 214, 172)
SIZE = (1400, 1020)


@dataclass
class InspectorControls:
    action: np.ndarray = field(
        default_factory=lambda: np.array([0, 0.5, 0], np.float32)
    )
    actor_control: bool = False
    paused: bool = True
    queued_steps: int = 0
    pending: bool = True
    selection: int | None = None
    factor: int = 0
    error_image: bool = False
    message: str = "Stage an action, then press . to advance one environment frame."

    def chosen_action(self, session):
        return session.actor_action() if self.actor_control else self.action.copy()

    def step(self, session):
        if session.done:
            self.paused = True
            self.queued_steps = 0
            self.message = "Episode ended. R resets this seed; N starts another track."
            return
        session.step(
            self.chosen_action(session),
            source="actor" if self.actor_control else "human",
        )
        self.pending = False
        self.selection = None
        self.queued_steps = max(0, self.queued_steps - 1)
        if session.done:
            self.paused = True
            self.queued_steps = 0

    def selected_transition(self, session):
        if not session.history:
            return None
        index = (
            self.selection if self.selection is not None else len(session.history) - 1
        )
        return session.history[index]

    def select(self, session, delta):
        if not session.history:
            return
        self.paused, self.pending, self.queued_steps = True, False, 0
        current = (
            self.selection if self.selection is not None else len(session.history) - 1
        )
        self.selection = int(np.clip(current + delta, 0, len(session.history) - 1))


class InspectorRenderer:
    def __init__(self):
        pygame.font.init()
        self.surface = pygame.Surface(SIZE)
        self.font = pygame.font.Font(None, 23)
        self.small = pygame.font.Font(None, 20)
        self.title = pygame.font.Font(None, 32)
        self.buttons = {}
        self.action_axes = []
        self.latent_rects = []
        self.history_rect = None

    def text(self, value, x, y, *, colour=TEXT, small=False, title=False):
        font = self.title if title else self.small if small else self.font
        self.surface.blit(font.render(str(value), True, colour), (x, y))

    def panel(self, rect, title):
        pygame.draw.rect(self.surface, PANEL, rect, border_radius=9)
        self.text(title, rect[0] + 14, rect[1] + 12)

    def button(self, key, label, x, width, *, active=False):
        rect = pygame.Rect(x, 65, width, 34)
        self.buttons[key] = rect
        pygame.draw.rect(self.surface, LINE if active else PANEL, rect, border_radius=5)
        self.text(label, x + 10, 73, small=True)

    def image(self, image, rect, placeholder="Awaiting environment step"):
        if image is None:
            pygame.draw.rect(self.surface, BACKGROUND, rect)
            rendered = self.small.render(placeholder, True, MUTED)
            self.surface.blit(
                rendered, rendered.get_rect(center=pygame.Rect(rect).center)
            )
            return
        image = np.ascontiguousarray(image, dtype=np.uint8)
        surface = pygame.image.frombuffer(image.tobytes(), image.shape[1::-1], "RGB")
        self.surface.blit(
            pygame.transform.scale(surface, pygame.Rect(rect).size), rect[:2]
        )

    def actor_bars(self, diagnostics, action, frame, *, historical=False):
        rect = pygame.Rect(20, 410, 464, 278)
        self.panel(
            rect,
            f"Actor at {'recorded' if historical else 'current'} decision state {frame}",
        )
        self.text(
            "Bar: mean   whisker: +/-1 SD   marker: supplied action",
            34,
            448,
            small=True,
            colour=MUTED,
        )
        self.action_axes = []
        for i, name in enumerate(("Steering", "Throttle", "Brake")):
            y = 488 + i * 61
            lower, upper = (
                float(diagnostics["action_lower"][i]),
                float(diagnostics["action_upper"][i]),
            )
            mean, std = (
                float(diagnostics["action_mean"][i]),
                float(diagnostics["action_std"][i]),
            )
            x1, x2 = 148, 453

            def position(value, lower=lower, upper=upper, left=x1, right=x2):
                return round(
                    left
                    + np.clip((value - lower) / (upper - lower), 0, 1) * (right - left)
                )

            self.text(name, 34, y - 7)
            self.text(
                f"{mean:+.2f}  SD {std:.2f}", 148, y - 24, small=True, colour=BLUE
            )
            pygame.draw.rect(
                self.surface, BACKGROUND, (x1, y, x2 - x1, 11), border_radius=3
            )
            zero, center = position(0), position(mean)
            pygame.draw.rect(
                self.surface,
                BLUE,
                (min(zero, center), y, max(2, abs(center - zero)), 11),
            )
            left, right = position(mean - std), position(mean + std)
            pygame.draw.line(self.surface, TEXT, (left, y + 5), (right, y + 5), 2)
            for x in (left, right):
                pygame.draw.line(self.surface, TEXT, (x, y - 1), (x, y + 12), 2)
            marker = position(action[i])
            pygame.draw.line(self.surface, ORANGE, (marker, y - 4), (marker, y + 16), 3)
            self.text(f"{lower:g}", x1, y + 16, small=True, colour=MUTED)
            self.text(f"{upper:g}", x2 - 10, y + 16, small=True, colour=MUTED)
            self.action_axes.append((
                i,
                pygame.Rect(x1, y - 10, x2 - x1 + 1, 37),
                lower,
                upper,
            ))

    def signals(self, forecast, after):
        self.panel((500, 410, 420, 278), "Next-state diagnostics")
        self.text("Signal", 514, 450, small=True, colour=MUTED)
        self.text("Prior", 650, 450, small=True, colour=BLUE)
        self.text("Posterior", 735, 450, small=True, colour=GREEN)
        self.text("Actual", 835, 450, small=True, colour=MUTED)
        rows = [
            ("Reward", "predicted_reward"),
            ("Continue", "predicted_continue"),
            ("Critic V", "value"),
            ("Target V", "target_value"),
        ]
        for index, (label, key) in enumerate(rows):
            y = 481 + index * 29
            self.text(label, 514, y, small=True)
            self.text(
                f"{float(forecast.diagnostics[key]):+.3g}",
                647,
                y,
                small=True,
                colour=BLUE,
            )
            if after is not None:
                self.text(
                    f"{float(after.after.diagnostics[key]):+.3g}",
                    735,
                    y,
                    small=True,
                    colour=GREEN,
                )
                observed = (
                    after.reward
                    if key == "predicted_reward"
                    else (
                        float(not (after.terminated or after.truncated))
                        if key == "predicted_continue"
                        else None
                    )
                )
                self.text(
                    "--" if observed is None else f"{observed:+.3g}",
                    835,
                    y,
                    small=True,
                    colour=MUTED,
                )
            else:
                self.text("--", 735, y, small=True, colour=MUTED)
                self.text("--", 835, y, small=True, colour=MUTED)
        y = 602
        self.text(
            "Next actor means: steer / throttle / brake",
            514,
            y,
            small=True,
            colour=MUTED,
        )
        for offset, (label, diagnostics, colour) in enumerate([
            ("Prior", forecast.diagnostics, BLUE),
            ("Post", None if after is None else after.after.diagnostics, GREEN),
        ]):
            values = (
                "--"
                if diagnostics is None
                else " / ".join(f"{v:+.2f}" for v in diagnostics["action_mean"])
            )
            self.text(
                f"{label}: {values}",
                514,
                y + 23 + offset * 22,
                small=True,
                colour=colour,
            )

    def returns(self, session, transition):
        self.panel((936, 410, 444, 278), "TD(lambda) targets under recorded actions")
        if transition is None:
            self.text("Record a transition, then press T.", 950, 453, colour=MUTED)
            return
        start = transition.before.index
        available = min(session.horizon, len(session.history) - start)
        result = session.comparisons.get(start)
        self.text(
            f"State {start} | {available}/{session.horizon} future steps available",
            950,
            451,
            small=True,
            colour=MUTED,
        )
        self.text(
            f"gamma {session.wm.cfg.discount:g} | lambda {session.wm.cfg.td_lambda:g}",
            950,
            477,
            small=True,
            colour=MUTED,
        )
        if result is None:
            self.text("T computes paired real/model targets.", 950, 520)
            self.text(
                "[ and ] select an earlier transition.",
                950,
                550,
                small=True,
                colour=MUTED,
            )
            return
        self.text("Value / target", 950, 510, small=True, colour=MUTED)
        self.text("Total", 1130, 510, small=True, colour=MUTED)
        self.text("Reward", 1204, 510, small=True, colour=MUTED)
        self.text("Bootstrap", 1285, 510, small=True, colour=MUTED)
        for i, (label, key, colour) in enumerate([
            ("Critic", "value", TEXT),
            ("Recorded", "recorded", GREEN),
            ("Imagined", "imagined", BLUE),
        ]):
            y = 540 + i * 32
            self.text(label, 950, y, small=True, colour=colour)
            self.text(f"{result[key]:+.2f}", 1126, y, small=True, colour=colour)
            if key != "value":
                self.text(
                    f"{result[key + '_reward']:+.2f}",
                    1204,
                    y,
                    small=True,
                    colour=colour,
                )
                self.text(
                    f"{result[key + '_bootstrap']:+.2f}",
                    1290,
                    y,
                    small=True,
                    colour=colour,
                )
        suffix = " (stale; T refreshes)" if result["steps"] != available else ""
        self.text(
            f"{result['steps']}-step target; critic bootstrap at tail{suffix}",
            950,
            649,
            small=True,
            colour=MUTED,
        )

    def latents(self, forecast, transition, selected):
        self.panel((20, 704, 900, 268), "Categorical latents at the next timestep")
        prior = softmax(np.asarray(forecast.state.prior[0]), axis=-1)
        posterior = (
            None
            if transition is None
            else softmax(np.asarray(transition.after.state.posterior[0]), axis=-1)
        )
        samples = [
            np.asarray(forecast.state.latent_sample[0]).reshape(prior.shape).argmax(-1)
        ]
        samples.append(
            None
            if transition is None
            else np
            .asarray(transition.after.state.latent_sample[0])
            .reshape(prior.shape)
            .argmax(-1)
        )
        self.latent_rects = []
        for index, (name, probs) in enumerate((
            ("Prior", prior),
            ("Posterior", posterior),
        )):
            x, y, extent = 34 + 258 * index, 751, 204
            self.text(name, x, 734, small=True, colour=MUTED)
            rect = pygame.Rect(x, y, extent, extent)
            self.latent_rects.append(rect)
            if probs is None:
                pygame.draw.rect(self.surface, BACKGROUND, rect)
                self.text(
                    "Awaiting observation", x + 9, y + 90, small=True, colour=MUTED
                )
                continue
            base = np.array(PANEL)
            colours = (
                base + (np.array(BLUE) - base) * (0.05 + 0.95 * probs[..., None])
            ).astype(np.uint8)
            tiles = pygame.surfarray.make_surface(np.transpose(colours, (1, 0, 2)))
            self.surface.blit(pygame.transform.scale(tiles, (extent, extent)), (x, y))
            rows, columns = probs.shape
            for row, choice in enumerate(samples[index]):
                cell = pygame.Rect(
                    x + round(choice * extent / columns),
                    y + round(row * extent / rows),
                    max(2, round(extent / columns)),
                    max(2, round(extent / rows)),
                )
                pygame.draw.rect(self.surface, ORANGE, cell, 1)
            pygame.draw.rect(
                self.surface,
                TEXT,
                (
                    x,
                    y + round(selected * extent / rows),
                    extent,
                    max(2, round(extent / rows)),
                ),
                1,
            )
        x = 557
        selected = min(selected, len(prior) - 1)
        self.text(f"Factor {selected} | {prior.shape[1]} classes", x, 750)
        entropy = lambda p: float(-(p * np.log(np.maximum(p, 1e-30))).sum())
        self.text(
            f"Prior entropy: {entropy(prior[selected]):.3f} nats", x, 786, small=True
        )
        if transition is not None:
            choice = samples[1][selected]
            self.text(
                f"Post entropy: {entropy(posterior[selected]):.3f} nats",
                x,
                813,
                small=True,
            )
            self.text(
                f"KL(q || p): {transition.factor_kl[selected]:.3f} nats",
                x,
                840,
                small=True,
            )
            self.text(f"Posterior sample: class {choice}", x, 867, small=True)
            self.text(
                f"Prior assigned it p={prior[selected, choice]:.4f}",
                x,
                894,
                small=True,
                colour=ORANGE,
            )
        self.text("Click a row to inspect a factor.", x, 930, small=True, colour=MUTED)

    def draw(self, session, controls, *, checkpoint_label=""):
        self.surface.fill(BACKGROUND)
        self.buttons = {}
        selected = controls.selected_transition(session)
        staged = controls.pending and controls.selection is None
        forecast = (
            session.preview(controls.chosen_action(session))
            if staged or selected is None
            else selected.forecast
        )
        transition = None if staged else selected
        before = session.current if transition is None else transition.before
        self.text("Dreamer driving inspector", 20, 15, title=True)
        self.text(
            f"{checkpoint_label} | seed {session.seed} | live state {session.current.index}",
            475,
            24,
            small=True,
            colour=MUTED,
        )
        for key, label, x, width, active in [
            (
                "pause",
                "Play" if controls.paused else "Pause",
                20,
                80,
                not controls.paused,
            ),
            ("step", "Step .", 110, 85, False),
            ("five", "+5", 205, 58, False),
            (
                "actor",
                "Actor drives" if controls.actor_control else "Human drives",
                273,
                134,
                controls.actor_control,
            ),
            (
                "imagine",
                "Open loop" if session.open_loop else "Observe each step",
                417,
                174,
                session.open_loop,
            ),
            ("correct", "Correct C", 601, 104, False),
            ("returns", "Lambda T", 715, 110, False),
            ("bookmark", "Bookmark B", 835, 128, False),
            ("save", "Save S", 973, 87, False),
            ("reset", "Reset R", 1070, 95, False),
        ]:
            self.button(key, label, x, width, active=active)
        description = f"{'STAGED' if transition is None else 'RECORDED'} transition {before.index} -> {before.index + 1}"
        if forecast.open_loop_start is not None:
            description += (
                f" | prior uncorrected since state {forecast.open_loop_start}"
            )
        self.text(
            description,
            20,
            111,
            small=True,
            colour=ORANGE if transition is None else MUTED,
        )
        panels = [
            ("Observation before action", before.image),
            ("Prior prediction of next image", forecast.diagnostics["reconstruction"]),
            (
                "Actual next image",
                None if transition is None else transition.after.image,
            ),
            (
                "Posterior reconstruction",
                None
                if transition is None
                else transition.after.diagnostics["reconstruction"],
            ),
        ]
        if controls.error_image and transition is not None:
            error = np.abs(
                forecast.diagnostics["reconstruction"].astype(np.int16)
                - transition.after.image.astype(np.int16)
            )
            panels[2] = (
                "Prediction error x4",
                np.minimum(255, error * 4).astype(np.uint8),
            )
        for i, (label, image) in enumerate(panels):
            x = 20 + i * 344
            self.panel((x, 140, 328, 254), label)
            self.image(image, (x + 60, 179, 208, 208))
        historical = controls.selection is not None
        actor_frame = selected.before if historical else session.current
        action = (
            selected.forecast.action if historical else controls.chosen_action(session)
        )
        self.actor_bars(
            actor_frame.diagnostics, action, actor_frame.index, historical=historical
        )
        self.signals(forecast, transition)
        self.returns(session, transition)
        self.latents(forecast, transition, controls.factor)
        self.panel((936, 704, 444, 268), "Controls and recording")
        lines = [
            "Arrows: steer / throttle / brake; release to neutral",
            "Click or drag actor axes to set precise actions",
            "Space: play/pause | .: step | 5: five steps",
            "A: actor control | I: open loop | C: correct prior",
            "[ / ]: inspect history | End: return to live state",
            "B: bookmark | Backspace: replay to selected state",
            "T: paired lambda targets | E: image error",
            "R: same track | N: new track | S: save | Esc: quit",
        ]
        for i, line in enumerate(lines):
            self.text(line, 950, 745 + 24 * i, small=True, colour=MUTED)
        self.history_rect = None
        if session.history:
            last = len(session.history) - 1
            selection = controls.selection if controls.selection is not None else last
            self.text(
                f"History: transition {selection} / {last}",
                950,
                935,
                small=True,
                colour=MUTED,
            )
            self.history_rect = pygame.Rect(950, 951, 413, 18)
            pygame.draw.line(self.surface, LINE, (950, 958), (1362, 958), 3)
            for item in session.bookmarks:
                if item["episode"] == len(session.archived):
                    x = 950 + round(min(item["frame"], last) / max(1, last) * 412)
                    pygame.draw.line(self.surface, GREEN, (x, 952), (x, 965), 2)
            x = 950 + round(selection / max(1, last) * 412)
            pygame.draw.circle(self.surface, BLUE, (x, 958), 5)
        message = controls.message
        while self.small.size(message)[0] > SIZE[0] - 40:
            message = message[:-4] + "..."
        self.text(message, 20, 987, small=True, colour=GREEN)
        return self.surface

    def action_at(self, position):
        for index, rect, lower, upper in self.action_axes:
            if rect.collidepoint(position):
                value = lower + np.clip(
                    (position[0] - rect.x) / max(1, rect.width - 1), 0, 1
                ) * (upper - lower)
                return index, float(np.round(value, 2))
        return None

    def factor_at(self, position, count):
        for rect in self.latent_rects:
            if rect.collidepoint(position):
                return min(count - 1, int((position[1] - rect.y) / rect.height * count))
        return None

    def history_at(self, position, count):
        if self.history_rect is not None and self.history_rect.collidepoint(position):
            fraction = (position[0] - self.history_rect.x) / (
                self.history_rect.width - 1
            )
            return round(fraction * (count - 1))
        return None


def handle_command(command, session, controls, save):
    if command in ("step", "five"):
        controls.selection = None
        controls.paused = True
        controls.queued_steps += 1 if command == "step" else 5
    elif command == "pause":
        controls.selection = None
        controls.queued_steps = 0
        controls.paused = not controls.paused
    elif command == "actor":
        controls.actor_control = not controls.actor_control
        controls.selection = None
        controls.pending = True
    elif command in ("imagine", "correct"):
        if command == "imagine":
            session.set_open_loop(not session.open_loop)
        else:
            session.correct()
        controls.selection = None
        controls.pending = True
    elif command == "returns":
        controls.paused, controls.queued_steps = True, 0
        transition = controls.selected_transition(session)
        if transition is not None:
            result = session.compare_returns(transition.before.index)
            controls.pending = False
            controls.message = f"Paired {result['steps']}-step targets use identical recorded actions; bootstraps remain critic estimates."
    elif command == "bookmark":
        frame = (
            session.current.index
            if controls.selection is None
            else session.history[controls.selection].before.index
        )
        item = session.bookmark(frame)
        controls.message = f"Bookmarked state {item['frame']}; Backspace restores the selected state by replay."
    elif command == "restore":
        if controls.selection is not None:
            frame = session.history[controls.selection].before.index
            session.restore(frame)
            (
                controls.selection,
                controls.pending,
                controls.paused,
                controls.queued_steps,
            ) = None, True, True, 0
            controls.message = (
                f"Restored state {frame}; previous branch retained in session export."
            )
    elif command in ("reset", "new"):
        session.reset(seed=session.seed + int(command == "new"))
        controls.selection, controls.pending, controls.paused, controls.queued_steps = (
            None,
            True,
            True,
            0,
        )
        controls.message = "Reset. Previous episode retained in session export."
    elif command == "save":
        controls.message = save()


def run_inspector(
    session, *, save, checkpoint_label="", fps=15, window_size=(1260, 918)
):
    pygame.init()
    window = pygame.display.set_mode(window_size, pygame.RESIZABLE)
    pygame.display.set_caption("Dreamer driving inspector")
    renderer, controls = InspectorRenderer(), InspectorControls()
    clock = pygame.time.Clock()
    running, dragging = True, None
    held = set()
    next_step = 0
    commands = {
        pygame.K_SPACE: "pause",
        pygame.K_PERIOD: "step",
        pygame.K_5: "five",
        pygame.K_a: "actor",
        pygame.K_i: "imagine",
        pygame.K_c: "correct",
        pygame.K_t: "returns",
        pygame.K_s: "save",
        pygame.K_b: "bookmark",
        pygame.K_BACKSPACE: "restore",
        pygame.K_r: "reset",
        pygame.K_n: "new",
    }
    keys = {
        pygame.K_LEFT: (0, -1),
        pygame.K_RIGHT: (0, 1),
        pygame.K_UP: (1, 1),
        pygame.K_DOWN: (2, 1),
    }
    try:
        while running:
            scale = min(window.get_width() / SIZE[0], window.get_height() / SIZE[1])
            offset = (
                (window.get_width() - SIZE[0] * scale) / 2,
                (window.get_height() - SIZE[1] * scale) / 2,
            )

            def logical(position, offset=offset, scale=scale):
                return (
                    (position[0] - offset[0]) / scale,
                    (position[1] - offset[1]) / scale,
                )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.WINDOWFOCUSLOST:
                    controls.paused, controls.queued_steps = True, 0
                    for key in held:
                        controls.action[keys[key][0]] = 0
                    if held:
                        controls.pending = True
                    held.clear()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in commands:
                        handle_command(commands[event.key], session, controls, save)
                    elif event.key in keys:
                        held.add(event.key)
                        index, value = keys[event.key]
                        if index == 0:
                            value = int(pygame.K_RIGHT in held) - int(
                                pygame.K_LEFT in held
                            )
                        controls.action[index] = value
                        controls.actor_control, controls.pending, controls.selection = (
                            False,
                            True,
                            None,
                        )
                    elif event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                        controls.select(
                            session, -1 if event.key == pygame.K_LEFTBRACKET else 1
                        )
                    elif event.key == pygame.K_END:
                        controls.selection, controls.pending = None, True
                    elif event.key == pygame.K_e:
                        controls.error_image = not controls.error_image
                elif event.type == pygame.KEYUP and event.key in keys:
                    held.discard(event.key)
                    index, value = keys[event.key]
                    if index == 0 or controls.action[index] == value:
                        controls.action[index] = (
                            int(pygame.K_RIGHT in held) - int(pygame.K_LEFT in held)
                            if index == 0
                            else 0
                        )
                        controls.pending = True
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = logical(event.pos)
                    for command, rect in renderer.buttons.items():
                        if rect.collidepoint(pos):
                            handle_command(command, session, controls, save)
                            break
                    else:
                        chosen = renderer.action_at(pos)
                        factor = renderer.factor_at(
                            pos, session.wm.cfg.num_latent_dists
                        )
                        history_index = renderer.history_at(pos, len(session.history))
                        if chosen is not None:
                            index, value = chosen
                            controls.action[index] = value
                            (
                                controls.actor_control,
                                controls.pending,
                                controls.selection,
                            ) = False, True, None
                            dragging = "action"
                        elif history_index is not None:
                            (
                                controls.selection,
                                controls.paused,
                                controls.pending,
                                controls.queued_steps,
                            ) = history_index, True, False, 0
                            dragging = "history"
                        elif factor is not None:
                            controls.factor = factor
                elif event.type == pygame.MOUSEBUTTONUP:
                    dragging = None
                elif event.type == pygame.MOUSEMOTION and dragging:
                    pos = logical(event.pos)
                    chosen = renderer.action_at(pos) if dragging == "action" else None
                    history_index = (
                        renderer.history_at(pos, len(session.history))
                        if dragging == "history"
                        else None
                    )
                    if chosen is not None:
                        index, value = chosen
                        controls.action[index] = value
                        controls.pending = True
                    elif history_index is not None:
                        controls.selection = history_index
            now = pygame.time.get_ticks()
            if running and (
                controls.queued_steps or (not controls.paused and now >= next_step)
            ):
                controls.step(session)
                next_step = pygame.time.get_ticks() + round(1000 / fps)
            surface = renderer.draw(
                session, controls, checkpoint_label=checkpoint_label
            )
            window.fill(BACKGROUND)
            size = (max(1, round(SIZE[0] * scale)), max(1, round(SIZE[1] * scale)))
            window.blit(pygame.transform.smoothscale(surface, size), offset)
            pygame.display.flip()
            clock.tick(30)
    finally:
        pygame.display.quit()
