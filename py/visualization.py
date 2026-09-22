"""
Interpolation schematic for the embedding-interpolation poster.

Draws the alpha line between two word vectors with the blend equation above it.
Writes figs/method_schematic.png next to this script, creating figs/ if needed.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- settings

ORIGINAL = "love"
TARGET = "hate"
ALPHA_MARKER = 0.62        # where the black dot sits on the line

GOLD = "#B89230"
LIGHT = "#FBE6A3"
INK = "#1a1a1a"
MUTED = "#6b6b6b"

OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ figure

fig, ax = plt.subplots(figsize=(8.6, 3.0), dpi=220)
ax.set_xlim(-0.12, 1.12)
ax.set_ylim(-0.52, 0.62)
ax.axis("off")

# the interpolation line
ax.plot([0, 1], [0, 0], color=GOLD, lw=4, solid_capstyle="round", zorder=1)

# endpoints
for x, label, sub in [(0.0, ORIGINAL, r"$v_{\mathrm{orig}}$"),
                      (1.0, TARGET, r"$v_{\mathrm{target}}$")]:
    ax.plot([x], [0], "o", ms=20, color="white", markeredgecolor=GOLD,
            markeredgewidth=3.5, zorder=3)
    ax.text(x, 0.16, label, ha="center", va="bottom", fontsize=19,
            fontweight="bold", color=INK)
    ax.text(x, -0.16, sub, ha="center", va="top", fontsize=16, color=MUTED)

# swept points along the segment
alphas = np.linspace(0.08, 0.92, 11)
ax.plot(alphas, np.zeros_like(alphas), "o", ms=8, color=GOLD, zorder=2)

# the current alpha marker
ax.plot([ALPHA_MARKER], [0], "o", ms=17, color=INK, zorder=4)
ax.annotate(r"$\alpha$", xy=(ALPHA_MARKER, 0), xytext=(ALPHA_MARKER, -0.34),
            ha="center", va="top", fontsize=20, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK, lw=2,
                            shrinkA=6, shrinkB=8))

# the blend equation, boxed
ax.text(0.5, 0.50,
        r"$e \;=\; (1-\alpha)\,v_{\mathrm{orig}} \;+\; \alpha\,v_{\mathrm{target}}$",
        ha="center", va="center", fontsize=21, color=INK,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=LIGHT,
                  edgecolor=GOLD, lw=2.5))

# ------------------------------------------------------------------- write

path = OUT / "method_schematic.png"
fig.tight_layout(pad=0.2)
fig.savefig(path, transparent=False, facecolor="white",
            bbox_inches="tight", pad_inches=0.12)
print(f"wrote {path}")