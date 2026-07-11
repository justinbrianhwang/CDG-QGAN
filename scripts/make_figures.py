"""Plot the result figures for the README and the paper.

Every number here is a MEASURED value copied from a results document, and the source is
named next to it. Nothing is illustrative. If a number changes, change it here and re-run —
do not redraw a figure by hand and do not let an image model draw anything with an axis.

Outputs: figures/*.png (README) and figures/*.svg (paper).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, GREEN, RED, GRAY = "#2C6FBB", "#E07B39", "#2E8B57", "#C0392B", "#8A8A8A"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})


def save(fig, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  figures/{name}.png  +  .svg")


# ---------------------------------------------------------------------------
# Figure A — the light-cone cliff.   Source: RESULTS_ceiling.md
# Maximum achievable |correlation| between a pair at graph distance d, after optimizing the
# circuit to maximize exactly that pair. The cliff must land at d = 2L.
# ---------------------------------------------------------------------------
def fig_lightcone_cliff() -> None:
    d = np.array([1, 2, 3, 4, 5, 6])
    L1 = np.array([0.991, 0.847, 0.012, 0.007, 0.010, 0.009])
    L2 = np.array([0.9999, 0.995, 0.997, 0.861, 0.009, 0.012])

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(d, L1, "o-", color=BLUE, lw=2, ms=7, label="L = 1  (reach 2L = 2)")
    ax.plot(d, L2, "s-", color=ORANGE, lw=2, ms=7, label="L = 2  (reach 2L = 4)")

    for x, c in ((2.5, BLUE), (4.5, ORANGE)):
        ax.axvline(x, color=c, ls=":", lw=1.4, alpha=0.8)
    ax.text(2.55, 0.62, "d = 2L", color=BLUE, fontsize=9, style="italic")
    ax.text(4.55, 0.62, "d = 2L", color=ORANGE, fontsize=9, style="italic")

    ax.axhspan(0, 0.05, color=GRAY, alpha=0.12)
    ax.text(0.85, 0.095, "finite-sample noise", ha="left", fontsize=8.5, color=GRAY)

    ax.set_xlabel("graph distance $d_G(u,v)$ between the pair")
    ax.set_ylabel("max achievable $|\\rho|$")
    ax.set_title("The light cone is exact: expressivity falls off a cliff at $d = 2L$",
                 fontsize=11.5, pad=12)
    ax.set_ylim(-0.03, 1.12)
    ax.set_xlim(0.7, 6.3)
    ax.set_xticks(d)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    ax.annotate("outside the cone,\n$\\mathrm{Cov}(x_u, x_v \\mid c) = 0$ exactly",
                xy=(3, 0.012), xytext=(2.15, 0.28), fontsize=9, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=1))
    save(fig, "fig_lightcone_cliff")


# ---------------------------------------------------------------------------
# Figure B — the alignment effect decays with depth.   Source: RESULTS_precheck.md
# Precheck z of the CDG against a 5,000-draw isomorphic-permutation null, on real MIMIC-IV.
# No training is involved. This is the gate.
# ---------------------------------------------------------------------------
def fig_alignment_decay() -> None:
    L = np.array([1, 2, 3])
    z = np.array([3.79, 1.41, 0.00])
    reach = ["48 / 120", "103 / 120", "120 / 120"]
    verdict = ["PASS", "weak", "meaningless"]
    pval = ["p < 0.0001 · 100th pct", "p = 0.038 · 86% reachable", "CDG $\\equiv$ permuted"]
    colors = [GREEN, ORANGE, RED]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.bar(L, np.maximum(z, 0.012), color=colors, width=0.5, zorder=3)
    ax.axhline(1.96, color="#444444", ls="--", lw=1.2, zorder=2)
    ax.text(3.45, 2.03, "p = 0.05", fontsize=8.5, color="#444444", ha="right")

    for x, y, v, p, c in zip(L, z, verdict, pval, colors):
        ax.text(x, y + 0.42, f"z = {y:+.2f}", ha="center", fontsize=11, fontweight="bold")
        ax.text(x, y + 0.14, f"{v} · {p}", ha="center", fontsize=8.5, color=c,
                fontweight="bold")

    ax.set_xticks(L)
    ax.set_xticklabels([f"$L = {i}$\nreach $2L = {2*i}$\n{r} pairs reachable"
                        for i, r in zip(L, reach)], fontsize=9)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_ylabel("alignment effect  $z$\n(CDG vs. 5,000-draw permutation null)")
    ax.set_title("Deeper circuits destroy the effect — exactly as the light cone predicts",
                 fontsize=11.5, pad=14)
    ax.set_ylim(0, 4.4)
    ax.set_xlim(0.5, 3.5)
    save(fig, "fig_alignment_decay")


# ---------------------------------------------------------------------------
# Figure C — joint expressivity ceiling.   Source: RESULTS_ceiling_joint.md
# GAN removed; the circuit is optimized directly on the full 120-pair pattern and scored on
# a fixed held-out draw (65,536 samples). Lower is better.
#
# FILL IN from the final ceiling_joint.py run before publishing.
# ---------------------------------------------------------------------------
JOINT = {
    "aligned\n(true CDG)": 0.0103,
    "rewired\n(degree-preserving)": 0.0401,
    "distmatched\n(distance-matched)": 0.0421,
    "permuted\n(isomorphic)": 0.0468,
    "no_entangle\n(RZZ removed)": 0.0508,
}
JOINT_FLOOR = 0.0609


def fig_joint_ceiling() -> None:
    names = [k for k, v in JOINT.items() if v is not None]
    vals = [JOINT[k] for k in names]
    if len(names) < 2:
        print("  (skipping fig_joint_ceiling — fill in JOINT from the ceiling_joint run)")
        return

    colors = [GREEN if "aligned" in n else (GRAY if "no_entangle" in n else BLUE)
              for n in names]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=colors, height=0.6, zorder=3)

    ax.axvline(JOINT_FLOOR, color=RED, ls="--", lw=1.4, zorder=4)
    ax.text(JOINT_FLOOR, y.min() - 0.62,
            f"floor = {JOINT_FLOOR:.4f}\na model that creates\nno dependency at all",
            color=RED, fontsize=8.5, va="top", ha="center", linespacing=1.4)

    for yy, v, n in zip(y, vals, names):
        gain = (JOINT_FLOOR - v) / JOINT_FLOOR * 100
        ax.text(v + 0.0012, yy, f"{v:.4f}   ({gain:.0f}% below floor)",
                va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("minimum reachable 120-pair dependency error  (lower = better)")
    ax.set_title("Same 19-edge entanglement budget — only the placement differs",
                 fontsize=11.5, pad=12)
    ax.set_xlim(0, JOINT_FLOOR * 1.42)
    ax.set_ylim(y.min() - 1.75, y.max() + 0.55)
    save(fig, "fig_joint_ceiling")


# ---------------------------------------------------------------------------
# Figure D — the confirmatory experiment.   Source: RESULTS_confirm.md
# A model TRAINED adversarially (pure WGAN-GP, copula + batch-aware critic), scored with the
# c-conditional metric the CDG is defined by. Lower is better.
# ---------------------------------------------------------------------------
CONFIRM = {                       # name: (120-pair error, sd, true-edge error)
    "aligned\n(true CDG)":        (0.0426, 0.0043, 0.1627),
    "no_entangle\n(RZZ removed)": (0.0647, 0.0003, 0.3644),
    "distmatched":                (0.0729, 0.0020, 0.3352),
    "rewired":                    (0.0749, 0.0051, 0.3521),
    "permuted\n(isomorphic)":     (0.0787, 0.0040, 0.3600),
}
CONFIRM_FLOOR = 0.0653
CONFIRM_FLOOR_EDGE = 0.3641


def fig_confirm() -> None:
    names = list(CONFIRM)
    v = np.array([CONFIRM[n][0] for n in names])
    sd = np.array([CONFIRM[n][1] for n in names])
    edge = np.array([CONFIRM[n][2] for n in names])
    colors = [GREEN if "aligned" in n else (GRAY if "no_entangle" in n else BLUE)
              for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    y = np.arange(len(names))[::-1]

    ax = axes[0]
    ax.barh(y, v, xerr=sd, color=colors, height=0.62, zorder=3,
            error_kw=dict(ecolor="#555555", lw=1.2, capsize=3))
    ax.axvline(CONFIRM_FLOOR, color=RED, ls="--", lw=1.4, zorder=4)
    ax.text(CONFIRM_FLOOR + 0.0013, y.min() - 0.62, f"floor = {CONFIRM_FLOOR:.4f}\n"
            "a model that creates\nno dependency at all",
            color=RED, fontsize=8.5, va="top", linespacing=1.4)
    for yy, x in zip(y, v):
        ax.text(x + 0.0035, yy, f"{x:.4f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 0.105)
    ax.set_ylim(y.min() - 1.9, y.max() + 0.6)
    ax.set_xlabel("conditional dependency error, all 120 pairs")
    ax.set_title("Trained model (pure WGAN-GP)", fontsize=11.5, pad=10)

    ax = axes[1]
    ax.barh(y, edge, color=colors, height=0.62, zorder=3)
    ax.axvline(CONFIRM_FLOOR_EDGE, color=RED, ls="--", lw=1.4, zorder=4)
    ax.text(CONFIRM_FLOOR_EDGE - 0.008, y.min() - 0.62,
            f"floor = {CONFIRM_FLOOR_EDGE:.4f}\nlearns nothing on\nthe true edges",
            color=RED, fontsize=8.5, va="top", ha="right", linespacing=1.4)
    for yy, x in zip(y, edge):
        ax.text(x + 0.012, yy, f"{x:.4f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(0, 0.50)
    ax.set_ylim(y.min() - 1.9, y.max() + 0.6)
    ax.set_xlabel("error on the 19 true edges")
    ax.set_title("Only the aligned circuit learns the true dependencies",
                 fontsize=11.5, pad=10)

    fig.suptitle("Same 19-edge budget · same critic · same training. Only the placement differs.",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.subplots_adjust(wspace=0.06, bottom=0.28)
    save(fig, "fig_confirm")


# ---------------------------------------------------------------------------
# Figure E — real MIMIC-IV: the training curves.   Source: results/wp2_probe.json
# The most important panel in the paper. Three models, one critic, one loss, 12,000 steps.
# no_entangle CONVERGES TO THE FLOOR AND STAYS THERE — Proposition D-2 as a curve, not an
# argument: strip the RZZ gates and ~2,000 head parameters cannot manufacture one dependence.
# ---------------------------------------------------------------------------
PROBE_STEPS = np.arange(1000, 12001, 1000)
PROBE = {
    "CDG (aligned)":  [0.1262, 0.1166, 0.0988, 0.0871, 0.0886, 0.0757,
                       0.0728, 0.0720, 0.0694, 0.0710, 0.0733, 0.0679],
    "permuted":       [0.1260, 0.1304, 0.1211, 0.1113, 0.1012, 0.0951,
                       0.0951, 0.0774, 0.0736, 0.0735, 0.0743, 0.0744],
    "no_entangle":    [0.1231, 0.1132, 0.1078, 0.1033, 0.0996, 0.0986,
                       0.0973, 0.0980, 0.0989, 0.0987, 0.0986, 0.0986],
}
PROBE_FLOOR = 0.0985
REAL_CEILING = 0.0437     # RESULTS_ceiling_real.md — reachable with the GAN removed
REAL_BOUND = 0.0331       # Corollary 1 — no L=1 CDG circuit can beat this


def fig_training_real() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    ax.axhspan(0, REAL_BOUND, color=GRAY, alpha=0.10, zorder=0)
    ax.axhline(REAL_BOUND, color="#555555", ls=":", lw=1.3, zorder=2)
    ax.text(12100, REAL_BOUND, "  Corollary 1 bound 0.0331\n  (72 pairs outside the L=1 cone)",
            fontsize=8, color="#555555", va="center", linespacing=1.4)

    ax.axhline(REAL_CEILING, color=GREEN, ls="-.", lw=1.4, zorder=2)
    ax.text(12100, REAL_CEILING, "  ceiling 0.0437\n  (GAN removed)",
            fontsize=8, color=GREEN, va="center", linespacing=1.4)

    ax.axhline(PROBE_FLOOR, color=RED, ls="--", lw=1.5, zorder=2)
    ax.text(12100, PROBE_FLOOR, "  floor 0.0985\n  (no dependency at all)",
            fontsize=8, color=RED, va="center", linespacing=1.4)

    for name, color, marker in (("CDG (aligned)", BLUE, "o"),
                                ("permuted", ORANGE, "s"),
                                ("no_entangle", GRAY, "^")):
        ax.plot(PROBE_STEPS, PROBE[name], marker + "-", color=color, lw=2, ms=5,
                label=name, zorder=3)

    ax.annotate("no_entangle lands exactly on the floor and stays:\nwith no RZZ gates, ~2,000 head parameters\n"
                "cannot create one dependence",
                xy=(9800, 0.0989), xytext=(6100, 0.1290), fontsize=8.5, color="#333333",
                linespacing=1.5, ha="left",
                arrowprops=dict(arrowstyle="->", color="#777777", lw=1,
                                connectionstyle="arc3,rad=-0.15"))
    ax.annotate("3,000 steps: sitting ON the floor.\nThe first WP-2 run stopped here.",
                xy=(3020, 0.0980), xytext=(3400, 0.0605), fontsize=8.5, color=RED,
                linespacing=1.5, ha="left",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))

    ax.set_xlabel("training step")
    ax.set_ylabel("conditional dependency error, all 120 pairs")
    ax.set_title("Real MIMIC-IV: only the CDG-wired circuit learns the clinical structure",
                 fontsize=12, pad=32)
    ax.set_xlim(500, 12500)
    ax.set_ylim(0.025, 0.140)
    ax.set_xticks(PROBE_STEPS[1::2])
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.005),
              handlelength=1.8, columnspacing=1.6)
    fig.subplots_adjust(right=0.74)
    save(fig, "fig_training_real")


# ---------------------------------------------------------------------------
# Figure F — real MIMIC-IV expressivity ceiling.   Source: RESULTS_ceiling_real.md
# GAN removed; circuit optimized directly on the real conditional pattern.
# ---------------------------------------------------------------------------
CEIL_REAL = {
    "aligned\n(CDG)": 0.0437,
    "permuted\n(isomorphic)": 0.0632,
    "distmatched": 0.0699,
    "rewired": 0.0730,
    "no_entangle": 0.0969,
}


def fig_ceiling_real() -> None:
    names = list(CEIL_REAL)
    vals = [CEIL_REAL[n] for n in names]
    colors = [GREEN if "aligned" in n else (GRAY if "no_entangle" in n else BLUE)
              for n in names]

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=colors, height=0.62, zorder=3)

    ax.axvspan(0, REAL_BOUND, color=GRAY, alpha=0.12, zorder=0)
    ax.axvline(REAL_BOUND, color="#555555", ls=":", lw=1.3, zorder=4)
    ax.axvline(PROBE_FLOOR, color=RED, ls="--", lw=1.4, zorder=4)

    ax.text(REAL_BOUND / 2, y.max() + 0.75, "unreachable\n(Corollary 1)", ha="center",
            fontsize=8, color="#555555", linespacing=1.3)
    ax.text(PROBE_FLOOR, y.min() - 0.65, f"floor {PROBE_FLOOR:.4f}\nno dependency at all",
            color=RED, fontsize=8.5, va="top", ha="center", linespacing=1.4)

    for yy, v in zip(y, vals):
        ax.text(v + 0.0016, yy, f"{v:.4f}", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("minimum reachable 120-pair conditional dependency error  (lower = better)")
    ax.set_title("Real MIMIC-IV · same 21-edge budget — only the placement differs",
                 fontsize=11.5, pad=22)
    ax.set_xlim(0, 0.113)
    ax.set_ylim(y.min() - 1.7, y.max() + 1.15)
    save(fig, "fig_ceiling_real")


if __name__ == "__main__":
    print("Writing figures to", OUT)
    fig_lightcone_cliff()
    fig_alignment_decay()
    fig_joint_ceiling()
    fig_confirm()
    fig_training_real()
    fig_ceiling_real()
