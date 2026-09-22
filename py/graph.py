"""
embedding_interpolation_3d.py

A low-dimensional cartoon of what embedding interpolation does: two word
embeddings sit as points in space, and a third point slides between them as
alpha goes 0 -> 1, following  v = (1 - alpha) * A + alpha * B.

Built for Manim Community Edition (v0.18+).

Render:
    manim -pqh embedding_interpolation_3d.py EmbeddingInterpolation
    #  -p preview,  q quality (l/m/h/k),  swap to -qk for 4k

Tweak WORD_A / WORD_B / positions / colors below to reuse for pizza-salad, etc.
"""

from manim import *
import numpy as np

# ----------------------------- config --------------------------------------
WORD_A   = "love"
WORD_B   = "hate"
A_POS    = np.array([-3.0, -1.5, -1.0])   # position of word A in axes coords
B_POS    = np.array([ 3.0,  1.5,  1.0])   # position of word B in axes coords
COLOR_A  = "#FF6E6E"                       # warm pole
COLOR_B  = "#34D9C4"                       # cool pole
ACCENT   = "#F5C451"                       # midpoint highlight
N_CLOUD  = 40                              # faint "other words" in the space
SWEEP_T  = 4.0                             # seconds for one A -> B sweep
# ---------------------------------------------------------------------------


class EmbeddingInterpolation(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            x_range=[-5, 5, 1], y_range=[-4, 4, 1], z_range=[-3, 3, 1],
            x_length=9, y_length=7, z_length=5,
            axis_config={"stroke_opacity": 0.35},
        )
        self.set_camera_orientation(phi=65 * DEGREES, theta=-50 * DEGREES, zoom=0.85)

        # --- faint cloud of other "words" so the two poles live in a space ---
        rng = np.random.default_rng(7)
        cloud = VGroup()
        for _ in range(N_CLOUD):
            p = rng.uniform([-4, -3, -2.5], [4, 3, 2.5])
            cloud.add(Dot3D(axes.c2p(*p), radius=0.04, color=GREY_B).set_opacity(0.35))

        # --- the two endpoint embeddings ---
        dotA = Dot3D(axes.c2p(*A_POS), radius=0.12, color=COLOR_A)
        dotB = Dot3D(axes.c2p(*B_POS), radius=0.12, color=COLOR_B)

        labelA = Text(WORD_A, color=COLOR_A, weight=BOLD).scale(0.55)
        labelB = Text(WORD_B, color=COLOR_B, weight=BOLD).scale(0.55)
        labelA.move_to(axes.c2p(*(A_POS + np.array([0, 0, 0.6]))))
        labelB.move_to(axes.c2p(*(B_POS + np.array([0, 0, 0.6]))))

        # --- dashed connector between the two words ---
        connector = DashedLine(
            dotA.get_center(), dotB.get_center(),
            color=GREY_A, stroke_width=2, dash_length=0.12,
        )

        # --- alpha tracker + the point that blends between A and B ----------
        alpha = ValueTracker(0.0)

        def blend_dot():
            a = alpha.get_value()
            pos = (1 - a) * A_POS + a * B_POS
            col = interpolate_color(ManimColor(COLOR_A), ManimColor(COLOR_B), a)
            return Dot3D(axes.c2p(*pos), radius=0.15, color=col)

        mover = always_redraw(blend_dot)

        # ------------------------------ HUD --------------------------------
        title = Text(
            "Blending one embedding into another",
            weight=BOLD,
        ).scale(0.5).to_corner(UL)
        subtitle = Text(
            "a 3-D cartoon of a 768-D space",
            color=GREY_B,
        ).scale(0.32).next_to(title, DOWN, aligned_edge=LEFT, buff=0.12)

        formula = Text(
            f"v = (1-a) \u00b7 {WORD_A} + a \u00b7 {WORD_B}",
            t2c={WORD_A: COLOR_A, WORD_B: COLOR_B},
        ).scale(0.5).to_corner(DL)

        alpha_read = VGroup(
            Text("a = ").scale(0.7),
            DecimalNumber(0, num_decimal_places=2, mob_class=Text).scale(0.7),
        ).arrange(RIGHT, buff=0.1).to_corner(DR)

        # -------------------------- choreography ---------------------------
        self.add(axes)
        self.play(FadeIn(cloud, run_time=1.2))

        self.add_fixed_in_frame_mobjects(title, subtitle)
        self.play(FadeIn(title), FadeIn(subtitle))

        self.add_fixed_orientation_mobjects(labelA, labelB)
        self.play(
            FadeIn(dotA), FadeIn(dotB),
            FadeIn(labelA), FadeIn(labelB),
        )

        self.play(Create(connector))

        self.add_fixed_in_frame_mobjects(formula, alpha_read)
        self.play(FadeIn(formula), FadeIn(alpha_read))

        self.add(mover)
        self.begin_ambient_camera_rotation(rate=0.04)

        # sweep to the halfway point and pause on the "not an average" beat
        self.play(alpha.animate.set_value(0.5), run_time=SWEEP_T / 2, rate_func=linear)

        mid_pos = 0.5 * (A_POS + B_POS)
        halo = Dot3D(axes.c2p(*mid_pos), radius=0.28, color=ACCENT).set_opacity(0.0)
        note = Text("halfway \u2260 half-meaning", color=ACCENT, weight=BOLD).scale(0.42)
        note.move_to(axes.c2p(*(mid_pos + np.array([0, 0, -0.9]))))
        self.add_fixed_orientation_mobjects(note)
        self.play(halo.animate.set_opacity(0.5), FadeIn(note))
        self.play(FadeOut(halo), run_time=0.6)
        self.wait(0.6)
        self.play(FadeOut(note), run_time=0.4)

        # continue to word B, then sweep back to A for a second look
        self.play(alpha.animate.set_value(1.0), run_time=SWEEP_T / 2, rate_func=linear)
        self.wait(0.3)
        self.play(alpha.animate.set_value(0.0), run_time=SWEEP_T, rate_func=smooth)

        self.stop_ambient_camera_rotation()
        self.wait(0.5)
        
        
