"""Manim scenes for output/final_demo.mp4 - title, comparison, architecture and
outro segments. Palette matches prism/viz_common.py exactly so these segments
read as part of the same product as the OpenCV-rendered stage videos.

Timings are tuned to hit exact frame counts at 30 fps (see tools/make_final_demo.py):
TitleScene 120f, CompareScene 150f, ArchScene 120f, OutroScene 210f.
"""
from __future__ import annotations
from manim import *

BG = "#0E1012"
PANEL = "#1A1D20"
EDGE = "#3A4046"
TEXT = "#E2E6E8"
TEXT_DIM = "#8C9296"
ACCENT = "#FFBE5A"
ACCENT2 = "#78E6C8"
OK = "#78E66E"
WARN = "#FFC83C"
BAD = "#EB3C3C"

FONT = "Arial"


def _card(w, h, color=PANEL, edge=EDGE, stroke=1.5):
    r = Rectangle(width=w, height=h, fill_color=color, fill_opacity=1.0,
                  stroke_color=edge, stroke_width=stroke)
    return r


class TitleScene(Scene):
    """0.0s - 4.0s : wordmark + tagline + capability strip."""

    def construct(self):
        self.camera.background_color = BG

        accent_bar = Rectangle(width=0.12, height=1.3, fill_color=ACCENT,
                               fill_opacity=1, stroke_width=0).shift(LEFT * 3.7)
        title = Text("prism", font=FONT, weight=BOLD, font_size=96, color=TEXT)
        title.next_to(accent_bar, RIGHT, buff=0.35)
        group = VGroup(accent_bar, title).move_to(UP * 0.55)

        tagline = Text("Camera-only navigation for GPS-denied off-road UGVs",
                       font=FONT, font_size=30, color=TEXT_DIM)
        tagline.next_to(group, DOWN, buff=0.45)

        chips_text = ["NO GPS", "NO LiDAR", "NO STEREO", "NO IMU"]
        chips = VGroup()
        for t in chips_text:
            lbl = Text(t, font=FONT, font_size=20, weight=BOLD, color=BG)
            bg = RoundedRectangle(corner_radius=0.08, width=lbl.width + 0.44,
                                  height=lbl.height + 0.32, fill_color=ACCENT2,
                                  fill_opacity=1, stroke_width=0)
            lbl.move_to(bg.get_center())
            chips.add(VGroup(bg, lbl))
        chips.arrange(RIGHT, buff=0.28).next_to(tagline, DOWN, buff=0.55)

        self.play(GrowFromEdge(accent_bar, DOWN), run_time=0.35)
        self.play(Write(title, run_time=1.1))
        self.play(FadeIn(tagline, shift=UP * 0.15), run_time=0.55)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.12) for c in chips], lag_ratio=0.12),
                  run_time=0.9)
        self.wait(0.9)


class CompareScene(Scene):
    """4.0s - 9.0s : the core "not a detector" argument, as a live comparison."""

    def construct(self):
        self.camera.background_color = BG

        header = Text("A DETECTOR ASKS ONE QUESTION. prism ASKS FOUR.",
                      font=FONT, font_size=30, weight=BOLD, color=TEXT)
        header.to_edge(UP, buff=0.65)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.5)

        rows = [
            ("what object is this?", "can THIS vehicle drive here?"),
            ("class + bounding box", "traversability + continuous risk"),
            ("silently confident", "explicit UNKNOWN when unsure"),
            ("this frame, right now", "rolls candidate actions forward first"),
        ]

        col_l = Text("YOLO-STYLE DETECTOR", font=FONT, font_size=22, weight=BOLD,
                     color=TEXT_DIM).move_to(LEFT * 3.6 + UP * 1.55)
        col_r = Text("prism", font=FONT, font_size=22, weight=BOLD,
                     color=ACCENT).move_to(RIGHT * 3.3 + UP * 1.55)
        self.play(FadeIn(col_l), FadeIn(col_r), run_time=0.4)

        line_group = VGroup()
        y0 = 0.85
        for i, (left, right) in enumerate(rows):
            y = y0 - i * 0.85
            lt = Text(left, font=FONT, font_size=24, color=TEXT_DIM)
            lt.move_to(LEFT * 3.6 + UP * y)
            arrow = Text("vs", font=FONT, font_size=18, color=EDGE).move_to(UP * y)
            rt = Text(right, font=FONT, font_size=24, color=TEXT, weight=BOLD)
            rt.move_to(RIGHT * 3.3 + UP * y)
            rule = Line(LEFT * 6.2 + UP * (y - 0.42), RIGHT * 6.2 + UP * (y - 0.42),
                       stroke_color=EDGE, stroke_width=1)
            self.play(FadeIn(lt, shift=RIGHT * 0.15),
                      FadeIn(arrow),
                      FadeIn(rt, shift=LEFT * 0.15),
                      Create(rule),
                      run_time=0.62)
            line_group.add(lt, arrow, rt, rule)

        self.wait(0.6)


class ArchScene(Scene):
    """9.0s - 13.0s : the pipeline flow, camera to decision."""

    def construct(self):
        self.camera.background_color = BG

        header = Text("ONE CAMERA -> A SAFETY-GATED DECISION",
                      font=FONT, font_size=28, weight=BOLD, color=TEXT)
        header.to_edge(UP, buff=0.55)
        self.play(FadeIn(header), run_time=0.4)

        stage_names = ["CAMERA", "DEPTH + TERRAIN", "TRAVERSABILITY\n+ CONFIDENCE",
                      "2.5D LOCAL MAP", "WORLD MODEL\n+ SUPERVISOR", "GO / SLOW /\nREROUTE / STOP"]
        colors = [TEXT_DIM, ACCENT, ACCENT2, OK, WARN, BAD]

        nodes = VGroup()
        n = len(stage_names)
        span = 12.0
        xs = [-span / 2 + span * i / (n - 1) for i in range(n)]
        for i, (name, col) in enumerate(zip(stage_names, colors)):
            lbl = Text(name, font=FONT, font_size=16, color=BG, weight=BOLD,
                      line_spacing=0.9)
            box = RoundedRectangle(corner_radius=0.1, width=1.95, height=1.05,
                                   fill_color=col, fill_opacity=1, stroke_width=0)
            if lbl.width > box.width - 0.22:
                lbl.scale_to_fit_width(box.width - 0.22)
            lbl.move_to(box.get_center())
            node = VGroup(box, lbl).move_to(RIGHT * xs[i])
            nodes.add(node)

        arrows = VGroup()
        for i in range(n - 1):
            a = Arrow(nodes[i].get_right(), nodes[i + 1].get_left(),
                     buff=0.06, stroke_width=3, color=EDGE, max_tip_length_to_length_ratio=0.28)
            arrows.add(a)

        rig = VGroup(nodes, arrows).move_to(ORIGIN + DOWN * 0.15)
        rig.scale_to_fit_width(13.0)

        for i in range(n):
            self.play(FadeIn(nodes[i], scale=0.85), run_time=0.32)
            if i < n - 1:
                self.play(GrowArrow(arrows[i]), run_time=0.22)

        self.wait(0.5)


class OutroScene(Scene):
    """53.0s - 60.0s : measured stats + honesty line + repo card."""

    def construct(self):
        self.camera.background_color = BG

        header = Text("MEASURED, NOT ASSERTED", font=FONT, font_size=30,
                      weight=BOLD, color=TEXT).shift(UP * 2.35)
        self.play(FadeIn(header, shift=DOWN * 0.1), run_time=0.45)

        stats = [
            ("45", "rendered demo videos"),
            ("9", "pipeline stages"),
            ("1", "RGB camera, no GPS/LiDAR/IMU"),
            ("0.055", "ground-plane fit residual (1/m)"),
        ]
        cards = VGroup()
        for val, label in stats:
            v = Text(val, font=FONT, font_size=46, weight=BOLD, color=ACCENT)
            l = Text(label, font=FONT, font_size=15, color=TEXT_DIM,
                    line_spacing=0.9).set_width(2.35)
            l.next_to(v, DOWN, buff=0.18)
            card = VGroup(v, l)
            cards.add(card)
        cards.arrange(RIGHT, buff=1.0).move_to(UP * 0.55)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cards], lag_ratio=0.15),
                  run_time=1.1)
        self.wait(0.5)

        honesty = Text(
            "Confidence is a confidence score, not a calibrated collision probability.",
            font=FONT, font_size=19, color=TEXT_DIM, slant=ITALIC)
        honesty.move_to(DOWN * 1.15)
        self.play(FadeIn(honesty), run_time=0.5)
        self.wait(0.6)

        wordmark = Text("prism", font=FONT, weight=BOLD, font_size=44, color=TEXT)
        repo = Text("github.com/X-DIABLO-X/prism", font=FONT, font_size=22,
                    color=ACCENT2)
        wm_group = VGroup(wordmark, repo).arrange(DOWN, buff=0.28).move_to(DOWN * 2.15)

        self.play(FadeOut(header), FadeOut(cards), FadeOut(honesty), run_time=0.5)
        self.play(Write(wm_group[0]), run_time=0.55)
        self.play(FadeIn(wm_group[1], shift=UP * 0.1), run_time=0.4)
        self.wait(1.0)
