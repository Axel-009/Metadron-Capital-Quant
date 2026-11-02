"""Corridor v2: participation s(Rm), corridor gamma ds/dRm, and leverage L = k·s·(Rm−rf)/σ² (cap 4.5x)."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "algorithm"))
import corridor as c

SURF, INK, INK2, GRID, SHADE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
rm = np.linspace(0.04, 0.18, 701)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(3, 1, figsize=(8.8, 10.4), sharex=True, facecolor=SURF,
                         gridspec_kw={"height_ratios": [1, 1, 1.5]})
for ax in axes:
    ax.set_facecolor(SURF); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.axvspan(c.R_LOW, c.R_HIGH, color=SHADE, zorder=0)
    ax.axhline(0, color=INK2, lw=0.9)

ax = axes[0]
ax.plot(rm, [c.participation(x) for x in rm], color=ORANGE, lw=2.2)
ax.set_title("Corridor participation  s(Rm) = 3t² − 2t³", loc="left", color=INK, fontsize=11.5)
ax.set_ylabel("s", rotation=0, labelpad=12); ax.set_ylim(-0.08, 1.15)
for x, lab in [(0.055, "DEFENSIVE\n0 long beta"), (0.0835, "DYNAMIC"), (0.15, "ROTATION\nfull vol-adj. Kelly")]:
    ax.text(x, 1.02 if x != 0.0835 else 0.62, lab, ha="center", color=INK2, fontsize=9.5, va="bottom")

ax = axes[1]
g = np.array([c.corridor_gamma(x) * 0.01 for x in rm])
ax.fill_between(rm, g, color=BLUE, alpha=0.12, lw=0); ax.plot(rm, g, color=BLUE, lw=2.2)
ax.scatter([0.095], [0.30], s=64, color=BLUE, edgecolor=SURF, linewidth=2, zorder=3)
ax.annotate("peak 0.30 per 1% at 9.5%  ·  area = 1", (0.095, 0.30), xytext=(12, -4),
            textcoords="offset points", color=INK, fontweight="bold")
ax.set_title("Corridor gamma  ds/dRm — the flattened bell", loc="left", color=INK, fontsize=11.5)
ax.set_ylabel("γ", rotation=0, labelpad=12); ax.set_ylim(-0.02, 0.38)

ax = axes[2]
for sig, col in [(0.12, AQUA), (0.15, BLUE), (0.20, ORANGE), (0.25, YELLOW), (0.35, MAGENTA)]:
    lev = [c.target_leverage(x, sig).leverage for x in rm]
    ax.plot(rm, lev, color=col, lw=2, label=f"σ = {sig:.0%}")
    if lev[-1] < c.MAX_GROSS_LEVERAGE - 0.05:
        ax.annotate(f"{sig:.0%}", (rm[-1], lev[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=9, va="center")
ax.axhline(c.MAX_GROSS_LEVERAGE, color=INK2, lw=1, ls=(0, (4, 3)))
ax.text(0.184, c.MAX_GROSS_LEVERAGE + 0.1, "max gross 4.5x  (σ 12%, 15% hit it)", color=INK2, fontsize=9.5, ha="right")
ax.set_title("Leverage  L = k · s(Rm) · (Rm − 4%) / σ²   (k = 1, full Kelly)", loc="left", color=INK, fontsize=11.5)
ax.set_ylabel("L", rotation=0, labelpad=12); ax.set_ylim(-0.2, 5.0)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), frameon=False, fontsize=9.5, ncol=1)
ax.set_xlabel("Annualised S&P 500 return Rm  (shaded: 7–12% corridor)")
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlim(0.04, 0.185)

fig.suptitle("Smart-beta corridor v2", x=0.075, ha="left", fontsize=14, color=INK, fontweight="bold")
fig.text(0.075, 0.94, "0 long beta below 7% · smooth ramp 7–12% · vol-adjusted full Kelly above 12% · cap 4.5x",
         color=INK2, fontsize=10)
fig.tight_layout(rect=(0, 0, 0.97, 0.935))
fig.savefig(__file__.replace(".py", ".png"), dpi=170, facecolor=SURF)
