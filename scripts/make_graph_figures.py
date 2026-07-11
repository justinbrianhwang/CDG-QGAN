"""Draw the reference-graph figures (aligned vs permuted, and the light cone).

Why these are plotted and not drawn by an image model
-----------------------------------------------------
The CDG is not a picture, it is an exact combinatorial object: 16 nodes, 19 edges, four
triangles, maximum degree 3, connected. Three rounds of image generation produced squares
instead of triangles, duplicate node labels, missing bridges, and features that do not exist
in the study (Age, Gender, BMI). Those are not styling mistakes — they contradict the paper.

The triangles in particular ARE the argument: they are why a strongly dependent pair stays at
graph distance 2, and therefore stays inside the light cone, even when its own edge is held
out. A 4-cycle has the same edge count and no common neighbor, so the pair falls to distance 3
and becomes unrepresentable. A figure that draws squares has drawn the wrong claim.

So the graph is constructed in code, and the two panels of the aligned/permuted figure are
drawn with *identical node positions* — enforced by construction, not by eye. That is the
whole point of the figure: the reader must not be able to tell the two graphs apart by shape.

Outputs: figures/fig_cdg_vs_permuted.{png,svg}, figures/fig_lightcone_graph.{png,svg}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, GREEN, RED, GRAY = "#2C6FBB", "#E07B39", "#2E8B57", "#C0392B", "#8A8A8A"
INK = "#333333"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

# --- The reference CDG ------------------------------------------------------
# Four clusters. Each cluster is a TRIANGLE plus one pendant node. Three bridges chain the
# clusters together. 16 nodes, 19 edges, max degree 3, connected.
CLUSTERS = [
    ("circulation", ["heart rate", "SBP", "DBP"], "respiratory rate"),
    ("respiration", ["SpO2", "temperature", "WBC"], "glucose"),
    ("electrolytes", ["sodium", "potassium", "chloride"], "bicarbonate"),
    ("renal / hematology", ["creatinine", "BUN", "hemoglobin"], "platelets"),
]
PENDANT_ANCHOR = {"respiratory rate": "DBP", "glucose": "WBC",
                  "bicarbonate": "chloride", "platelets": "hemoglobin"}
BRIDGES = [("respiratory rate", "SpO2"), ("glucose", "sodium"),
           ("bicarbonate", "creatinine")]


def reference_cdg() -> nx.Graph:
    G = nx.Graph()
    for _, tri, pend in CLUSTERS:
        G.add_edges_from([(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])])
        G.add_edge(pend, PENDANT_ANCHOR[pend])
    G.add_edges_from(BRIDGES)
    return G


# Explicit geometry. The four clusters are chained C1 -> C2 -> C3 -> C4 around the figure, so
# the three bridges are short, axis-aligned, and never cross an edge. Label anchors are set
# per node (outward from its cluster) so no two labels collide.
POS = {
    # C1 circulation (top left) — triangle, pendant points right toward C2
    "heart rate":       (-2.95,  1.85),
    "SBP":              (-2.95,  0.75),
    "DBP":              (-1.95,  1.30),
    "respiratory rate": (-0.95,  1.30),
    # C2 respiration (top right) — SpO2 receives the bridge; pendant points down toward C3
    "SpO2":             ( 1.05,  1.30),
    "temperature":      ( 2.10,  1.85),
    "WBC":              ( 2.10,  0.75),
    "glucose":          ( 2.70, -0.15),
    # C3 electrolytes (bottom right) — sodium receives the bridge; pendant points left to C4
    "sodium":           ( 2.70, -1.05),
    "potassium":        ( 3.05, -2.10),
    "chloride":         ( 1.90, -1.85),
    "bicarbonate":      ( 0.85, -1.85),
    # C4 renal / hematology (bottom left) — creatinine receives the bridge
    "creatinine":       (-1.15, -1.85),
    "BUN":              (-2.20, -1.25),
    "hemoglobin":       (-2.20, -2.35),
    "platelets":        (-3.25, -2.35),
}
# (dx, dy, ha, va) for the text label of each node
LAB = {
    "heart rate":       (0, .28, "center", "bottom"),
    "SBP":              (-.22, -.10, "right", "center"),
    "DBP":              (0, -.28, "center", "top"),
    "respiratory rate": (0, .28, "center", "bottom"),
    "SpO2":             (0, .28, "center", "bottom"),
    "temperature":      (0, .28, "center", "bottom"),
    "WBC":              (.24, -.02, "left", "center"),
    "glucose":          (.24, 0, "left", "center"),
    "sodium":           (.24, 0, "left", "center"),
    "potassium":        (.22, -.12, "left", "center"),
    "chloride":         (0, -.28, "center", "top"),
    "bicarbonate":      (0, -.28, "center", "top"),
    "creatinine":       (0, -.28, "center", "top"),
    "BUN":              (-.24, .04, "right", "center"),
    "hemoglobin":       (0, -.28, "center", "top"),
    "platelets":        (0, -.28, "center", "top"),
}


def layout(G: nx.Graph) -> dict:
    return dict(POS)


def draw_graph(ax, G, pos, labels, accent, highlight=None, hl_style="solid", hl_label=""):
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#B9B9B9", width=1.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="white", edgecolors="#8A8A8A",
                           linewidths=1.3, node_size=430)
    for n, (x, y) in pos.items():
        dx, dy, ha, va = LAB[n]
        ax.text(x + dx, y + dy, labels[n], ha=ha, va=va, fontsize=7.6, color=INK)

    if highlight:
        u, v = highlight
        if hl_style == "solid":
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, edge_color=accent,
                                   width=4.0)
        else:  # the pair is NOT an edge — draw the shortest path through the graph
            path = nx.shortest_path(G, u, v)
            pe = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=pe, ax=ax, edge_color=accent, width=3.4,
                                   style=(0, (2, 1.6)))
        nx.draw_networkx_nodes(G, pos, nodelist=[u, v], ax=ax, node_color=accent,
                               edgecolors=accent, node_size=470, alpha=0.30)
        nx.draw_networkx_nodes(G, pos, nodelist=[u, v], ax=ax, node_color="none",
                               edgecolors=accent, linewidths=2.2, node_size=470)
    ax.set_axis_off()


def fig_cdg_vs_permuted() -> None:
    G = reference_cdg()
    assert G.number_of_nodes() == 16 and G.number_of_edges() == 19
    assert max(d for _, d in G.degree()) == 3 and nx.is_connected(G)
    assert sum(nx.triangles(G).values()) // 3 == 4

    pos = layout(G)
    ident = {n: n for n in G}

    # The permutation is isomorphic: identical graph, relabelled. Chosen so that the
    # creatinine-BUN pair (|rho| = 0.66) ends up far apart.
    names = list(G.nodes())
    rng = np.random.default_rng(3)
    for _ in range(400):
        perm = list(rng.permutation(names))          # slot -> clinical name
        lab = dict(zip(names, perm))
        inv = {v: k for k, v in lab.items()}
        if nx.shortest_path_length(G, inv["creatinine"], inv["BUN"]) >= 4:
            break

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6))

    draw_graph(axes[0], G, pos, ident, GREEN, highlight=("creatinine", "BUN"),
               hl_style="solid")
    axes[0].set_title("Aligned (CDG)", fontsize=14, fontweight="bold", color=GREEN, pad=14)
    axes[0].text(0.5, 1.005, "clinically related variables share a triangle",
                 transform=axes[0].transAxes, ha="center", fontsize=9, color=INK)
    axes[0].text(0.5, -0.03, "creatinine — BUN   $|\\rho| = 0.66$\nadjacent · inside the "
                 "light cone  ✓", transform=axes[0].transAxes, ha="center", va="top",
                 fontsize=9.5, color=GREEN, fontweight="bold", linespacing=1.5)

    cu, cv = inv["creatinine"], inv["BUN"]
    d = nx.shortest_path_length(G, cu, cv)
    draw_graph(axes[1], G, pos, lab, RED, highlight=(cu, cv), hl_style="dashed")
    axes[1].set_title("Permuted (isomorphic)", fontsize=14, fontweight="bold", color=RED,
                      pad=14)
    axes[1].text(0.5, 1.005, "identical graph, clinical labels scrambled",
                 transform=axes[1].transAxes, ha="center", fontsize=9, color=INK)
    axes[1].text(0.5, -0.03, f"creatinine — BUN   $|\\rho| = 0.66$\ndistance {d} · outside "
                 "the light cone  ✗  unrepresentable", transform=axes[1].transAxes,
                 ha="center", va="top", fontsize=9.5, color=RED, fontweight="bold",
                 linespacing=1.5)

    for ax in axes:
        ax.set_xlim(-3.8, 3.7)
        ax.set_ylim(-3.0, 2.4)

    fig.text(0.5, 0.005,
             "Same nodes.  Same edges.  Same degree sequence.  Same triangles.\n"
             "The ONLY difference is which clinical pair sits where.",
             ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold",
             linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.7", fc="#F4F4F4", ec="#CCCCCC"))
    fig.subplots_adjust(bottom=0.20, wspace=0.02)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"fig_cdg_vs_permuted.{ext}")
    plt.close(fig)
    print("  figures/fig_cdg_vs_permuted.png  +  .svg"
          f"   (creatinine–BUN: distance 1 aligned, distance {d} permuted)")


def fig_lightcone_graph() -> None:
    """The same graph, showing how the cone widens from L=1 to L=2.

    Honesty note: this schematic uses the synthetic teacher graph (19 edges, diameter 11), so
    L=3 does NOT reach every pair *here*. The claim that L=3 makes the CDG identical to a
    permuted graph is about the REAL CDG (23 edges, diameter 5), and it is carried by
    fig_alignment_decay, which plots the measured precheck z. Do not restate it on this
    figure — it would be a claim the drawn graph does not support.
    """
    G = reference_cdg()
    pos = layout(G)
    u = "creatinine"

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
    for ax, L, accent in ((axes[0], 1, BLUE), (axes[1], 2, RED)):
        reach = 2 * L
        dist = nx.single_source_shortest_path_length(G, u)
        inside = [n for n, d in dist.items() if d <= reach]
        outside = [n for n in G if n not in inside]

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#C8C8C8", width=1.5)
        nx.draw_networkx_nodes(G, pos, nodelist=inside, ax=ax, node_color=accent,
                               alpha=0.22, edgecolors=accent, linewidths=1.4, node_size=470)
        nx.draw_networkx_nodes(G, pos, nodelist=outside, ax=ax, node_color="white",
                               edgecolors="#BBBBBB", linewidths=1.2, node_size=430)
        nx.draw_networkx_nodes(G, pos, nodelist=[u], ax=ax, node_color=accent,
                               edgecolors=accent, node_size=520)
        for n, (x, y) in pos.items():
            dx, dy, ha, va = LAB[n]
            ax.text(x + dx, y + dy, n, ha=ha, va=va, fontsize=7.6,
                    color=INK if n in inside else "#AFAFAF",
                    fontweight="bold" if n == u else "normal")
        ax.set_axis_off()
        ax.set_xlim(-3.8, 3.7)
        ax.set_ylim(-3.0, 2.4)
        ax.set_title(f"$L = {L}$   (reach $2L = {reach}$)", fontsize=13, fontweight="bold",
                     color=accent, pad=12)
        ax.text(0.5, 0.015, f"{len(inside)} of 16 features reachable from creatinine",
                transform=ax.transAxes, ha="center", fontsize=9.5, color=accent,
                fontweight="bold")

    axes[0].text(0.5, -0.085, "inside the cone: $\\mathrm{Cov}(x_u, x_v \\mid c)$ can be "
                 "nonzero\noutside it: exactly zero, whatever the parameters",
                 transform=axes[0].transAxes, ha="center", fontsize=9.5, color=INK,
                 linespacing=1.5)
    axes[1].text(0.5, -0.085, "the cone widens → fewer pairs are forced to zero\n"
                 "→ the topology constrains less, and discriminates less",
                 transform=axes[1].transAxes, ha="center", fontsize=9.5, color=RED,
                 fontweight="bold", linespacing=1.5)

    fig.text(0.5, 0.005,
             "Shallow is not a compromise. It is the only regime in which the topology "
             "carries information.\n"
             "On the real CDG (16 nodes, 23 edges, diameter 5) the effect is already gone at "
             "$L=2$ (z = +0.66) and null by definition at $L=3$.",
             ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="bold",
             linespacing=1.6)
    fig.subplots_adjust(bottom=0.20, wspace=0.02)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"fig_lightcone_graph.{ext}")
    plt.close(fig)
    print("  figures/fig_lightcone_graph.png  +  .svg")


if __name__ == "__main__":
    G = reference_cdg()
    print(f"Reference CDG: {G.number_of_nodes()} nodes · {G.number_of_edges()} edges · "
          f"max degree {max(d for _, d in G.degree())} · "
          f"{sum(nx.triangles(G).values()) // 3} triangles · "
          f"connected={nx.is_connected(G)} · diameter={nx.diameter(G)}")
    fig_cdg_vs_permuted()
    fig_lightcone_graph()
