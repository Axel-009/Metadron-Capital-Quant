import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
s2, h = 0.15**2, 0.0732
g_you = 0.05 / (0.454 * s2)          # 4.89
rm = np.linspace(0.07, 0.12, 201)
gamma = (rm - h) / s2
beta = gamma / g_you

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.4), sharex=True, facecolor=SURF)
panels = [
    (axes[0], gamma, BLUE, "Market-implied risk aversion  γ(Rm) = (Rm − h) / σ²",
     "γ", [(0.07, "−0.14"), (0.095, "0.97"), (0.12, "2.08")]),
    (axes[1], beta, ORANGE, f"Your exposure  β(Rm) = γ(Rm) / γ_you   (γ_you = {g_you:.2f})",
     "β", [(0.07, "−0.029"), (0.095, "0.198"), (0.12, "0.425")]),
]
for ax, y, c, title, ylab, pts in panels:
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.axhline(0, color=INK2, lw=0.9)
    ax.axvline(h, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.plot(rm, y, color=c, lw=2, solid_capstyle="round")
    for x, lab in pts:
        yy = np.interp(x, rm, y)
        ax.scatter([x], [yy], s=64, color=c, edgecolor=SURF, linewidth=2, zorder=3)
        ha = "left" if x < 0.12 else "right"
        dy = -18 if x == 0.07 else 8
        ax.annotate(lab, (x, yy), xytext=(8 if ha == "left" else -8, dy),
                    textcoords="offset points", ha=ha, color=INK, fontsize=10.5, fontweight="bold")
    ax.set_title(title, loc="left", color=INK, fontsize=11.5, pad=10)
    ax.set_ylabel(ylab, rotation=0, labelpad=14, fontsize=12)
    lo, hi = y.min(), y.max(); pad = (hi - lo) * 0.18
    ax.set_ylim(lo - pad, hi + pad)
    ax.text(h, hi + pad * 0.55, "  h = 7.32%  (zero crossing)", color=INK2, fontsize=9.5, va="center")
axes[1].set_xlabel("Expected market return Rm  (7–12% corridor)", fontsize=10.5)
axes[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
axes[1].set_xlim(0.0685, 0.1215)
fig.suptitle("Gamma and beta across the 7–12% corridor", x=0.075, ha="left",
             fontsize=14, color=INK, fontweight="bold")
fig.text(0.075, 0.925, "σ = 15%  ·  hurdle h ≈ 7.32%  ·  γ_you ≈ 4.89  ·  β = 1/γ_you ≈ 20% of full Kelly",
         color=INK2, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.915))
fig.savefig(__file__.replace(".py", ".png"), dpi=180, facecolor=SURF)
