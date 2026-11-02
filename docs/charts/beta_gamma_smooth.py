"""Smooth beta curve and its gamma (dβ/dRm) across the 7–12% corridor."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RL, RH, BL, BH = 0.07, 0.12, -0.029, 0.425
D = BH - BL                                   # 0.454 total beta change

rm = np.linspace(0.05, 0.14, 721)
t = np.clip((rm - RL) / (RH - RL), 0, 1)
beta_smooth = BL + D * (3 * t**2 - 2 * t**3)                   # smoothstep S-curve
gamma_smooth = np.where((rm > RL) & (rm < RH), D * 6 * t * (1 - t) / (RH - RL), 0.0) / 100  # per 1% Rm
beta_lin = BL + D * t
gamma_lin = np.where((rm >= RL) & (rm <= RH), D / (RH - RL), 0.0) / 100

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.6), sharex=True, facecolor=SURF)
for ax in axes:
    ax.set_facecolor(SURF); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.axvspan(RL, RH, color="#f0efec", zorder=0)
    ax.axhline(0, color=INK2, lw=0.9)

ax = axes[0]
ax.plot(rm, beta_lin, color=INK2, lw=1.4, ls=(0, (4, 3)), label="linear (current code)")
ax.plot(rm, beta_smooth, color=ORANGE, lw=2.2, label="smooth S-curve")
for x, y, lab, dy in [(RL, BL, "−0.029", -18), (0.095, BL + D / 2, "0.198", 10), (RH, BH, "0.425", 10)]:
    ax.scatter([x], [y], s=64, color=ORANGE, edgecolor=SURF, linewidth=2, zorder=3)
    ax.annotate(lab, (x, y), xytext=(8, dy), textcoords="offset points", color=INK, fontweight="bold")
ax.set_title("β(Rm) — exposure across the corridor", loc="left", color=INK, fontsize=11.5, pad=10)
ax.set_ylabel("β", rotation=0, labelpad=14, fontsize=12)
ax.set_ylim(-0.12, 0.52)
ax.legend(loc="upper left", frameon=False, fontsize=9.5)

ax = axes[1]
ax.plot(rm, gamma_lin, color=INK2, lw=1.4, ls=(0, (4, 3)), label="linear → flat box (0.091 per 1%)")
ax.fill_between(rm, gamma_smooth, color=BLUE, alpha=0.12, lw=0)
ax.plot(rm, gamma_smooth, color=BLUE, lw=2.2, label="smooth → flattened bell")
peak = D * 1.5 / (RH - RL) / 100
ax.scatter([0.095], [peak], s=64, color=BLUE, edgecolor=SURF, linewidth=2, zorder=3)
ax.annotate(f"peak {peak:.3f} per 1% at 9.5%", (0.095, peak), xytext=(12, 4),
            textcoords="offset points", color=INK, fontweight="bold")
ax.text(0.095, 0.012, "area under curve = 0.454 (= 0.425 − (−0.029))", ha="center", color=INK2, fontsize=9.5)
ax.set_title("γ(Rm) = dβ/dRm — gamma, change in β per 1% of Rm", loc="left", color=INK, fontsize=11.5, pad=10)
ax.set_ylabel("γ", rotation=0, labelpad=14, fontsize=12)
ax.set_ylim(-0.01, 0.17)
ax.legend(loc="upper left", frameon=False, fontsize=9.5)
ax.set_xlabel("Expected market return Rm  (shaded: 7–12% corridor)")
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlim(0.05, 0.14)

fig.suptitle("Smooth beta curve and its gamma", x=0.075, ha="left", fontsize=14, color=INK, fontweight="bold")
fig.text(0.075, 0.925, "β flat outside the corridor, S-shaped inside; γ = slope of β — zero outside, bell-shaped inside",
         color=INK2, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.915))
fig.savefig(__file__.replace(".py", ".png"), dpi=180, facecolor=SURF)
