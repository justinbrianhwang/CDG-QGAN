# Figure prompts for the README

## Revision round 1 — what came back and what to fix

| Figure | Verdict |
|---|---|
| 4 — local heads | **accept as-is** |
| 6 — the MAP trap | **accept as-is** |
| 2 — light cone | usable, but redraw on the real CDG rather than a square lattice |
| 1 — pipeline | **fix**: the CDG has only 15 nodes and node `13` appears twice; heads all share the subscript `u` |
| 3 — aligned vs permuted | **redraw**: it is drawn as a ring, so there are no triangles at all — yet the caption claims "same triangles". A ring is also one of our actual control graphs, so using it here is actively confusing. Also `pH` is not one of our 16 features and `respiratory rate` is missing |
| 5 — data provenance | **fix**: the arrow out of `hosp / labevents` goes to the condition vector. It must go to the 10 labs. The condition vector is fed by `patients` / `admissions` / `icustays` |

### The 16 features — use exactly these names, no others

`heart rate`, `SBP`, `DBP`, `respiratory rate`, `SpO2`, `temperature` (6 vitals) ·
`WBC`, `glucose`, `sodium`, `potassium`, `chloride`, `bicarbonate`, `creatinine`, `BUN`,
`hemoglobin`, `platelets` (10 labs).

There is **no pH**. There is **no MAP** among the generated features (MAP is evaluation-only;
see Figure 6).

### The reference graph — use this exact graph in Figures 1, 2, and 3

**16 nodes, 19 edges, maximum degree 3.** Four clusters of four. Each cluster is a
**triangle plus one pendant node**, and the clusters are chained by three bridges.

**The triangles are the entire point.** They are why a strongly-dependent pair stays at
distance 2 — and therefore stays *representable* — even when its own edge is held out. A
4-cycle has the same number of edges but no common neighbor, and it would destroy the
argument. Do not draw squares. Do not draw a ring. Do not draw a lattice.

Complete edge list — draw exactly these 19 edges and no others:

```
CLUSTER 1  "circulation"        triangle: heart rate — SBP
                                          SBP        — DBP
                                          DBP        — heart rate
                                pendant:  DBP        — respiratory rate

CLUSTER 2  "respiration"        triangle: SpO2        — temperature
                                          temperature — WBC
                                          WBC         — SpO2
                                pendant:  WBC         — glucose

CLUSTER 3  "electrolytes"       triangle: sodium    — potassium
                                          potassium — chloride
                                          chloride  — sodium
                                pendant:  chloride  — bicarbonate

CLUSTER 4  "renal / hematology" triangle: creatinine — BUN
                                          BUN        — hemoglobin
                                          hemoglobin — creatinine
                                pendant:  hemoglobin — platelets

BRIDGES (these connect the clusters — the graph must be CONNECTED)
           respiratory rate — SpO2          (cluster 1 → 2)
           glucose          — sodium        (cluster 2 → 3)
           bicarbonate      — creatinine    (cluster 3 → 4)
```

Label the nodes with the **clinical names above**, not with numbers. If numbers are needed
for the qubit wires in Figure 1, put them as a small secondary label (`q_1` … `q_16`).

Checklist before you export: 16 distinct nodes · 19 edges · 4 visible triangles · 3 bridges ·
the graph is connected end to end · no node label appears twice · no node has degree > 3.

---

## Ground rules

**Do NOT generate figures that carry data with an image model.** Bar heights, curves, and
tables must come from the actual numbers, or the figure is wrong. Those are plotted with
matplotlib from the result files (see "Plotted, not generated" at the bottom).

Image models are for **conceptual diagrams** only: the pipeline, the circuit, the light
cone, and the aligned-vs-permuted contrast.

**Shared style directive** — paste this into every prompt so the four figures look like one
set:

> Style: clean scientific diagram for a machine-learning paper. Flat vector look, no 3D, no
> gradients, no drop shadows, no glow, no photorealism, no stock-photo elements. White
> background. Thin (1–1.5px) dark-gray strokes. Restrained palette: one blue (#2C6FBB) for
> the quantum/circuit side, one warm orange (#E07B39) for the clinical/data side, one green
> (#2E8B57) for "aligned/correct", one red (#C0392B) for "misaligned/unreachable", neutral
> grays for everything else. Sans-serif labels (Inter or Helvetica), all text horizontal and
> legible. Generous whitespace. Think Distill.pub or a Nature Methods schematic, not a
> marketing graphic. No decorative icons. Every element must be labeled with the exact text
> I give — do not invent, translate, or paraphrase any label.

---

## Figure 1 — The pipeline (the most important one)

Aspect ratio 16:9, wide. A single left-to-right flow in five stages, with a feedback arrow.

> Draw a horizontal five-stage pipeline diagram, left to right, connected by thin arrows.
>
> **Stage 1 — "MIMIC-IV ICU cohort"** (orange). A small table icon labeled
> `24h landmark cohort` with a caption underneath reading `n ≈ 48,560 · 16 clinical features`.
> Under it, a small tag: `condition c = (mortality, age, sex, ICU type)`.
>
> **Stage 2 — "Clinical Dependency Graph (CDG)"** (orange). A 16-node undirected graph,
> sparse, maximum degree 3, showing visible clusters. Label four clusters with small text:
> `circulation`, `electrolytes`, `renal`, `hematology`. Caption underneath:
> `residualize on c → nonparanormal → graphical lasso → stability selection`.
>
> **Stage 3 — "Quantum circuit (the graph IS the topology)"** (blue). A quantum circuit with
> 16 horizontal qubit wires. On each wire, in order: a box labeled `RY(a·z_u + b·y + c)`,
> then a box labeled `RZ`. Then a set of two-qubit vertical RZZ links — and these RZZ links
> must connect exactly the same pairs of wires as the edges of the graph in Stage 2 (draw a
> faint dotted guideline from the Stage-2 graph to the Stage-3 RZZ links to make that
> correspondence explicit). After the RZZ links, another column of single-qubit boxes labeled
> `RX`, `RY`. At the right end of each wire, a measurement symbol labeled `⟨Z_u⟩`. Big caption
> under the whole stage: `1 feature = 1 qubit · RZZ only on CDG edges · depth L=1`.
>
> **Stage 4 — "Local heads"** (gray). 16 small separate boxes, one per wire, each labeled
> `h_u(q_u, c)`. They must be visibly *separate and unconnected* — no lines between them.
> Caption: `1-D map per feature. Cannot mix features.`
>
> **Stage 5 — "Synthetic ICU data"** (orange). A small table icon.
>
> Finally, below Stage 5, draw a box labeled `Critic D(x, c)` with an arrow going back to
> Stage 3, labeled `WGAN-GP`.
>
> [+ shared style directive]

**What must be exactly right:** the RZZ links in stage 3 must match the graph edges in stage
2 — that visual identity IS the paper's idea. And the 16 heads must look disconnected from
each other; that's the D-2 proposition.

---

## Figure 2 — The light cone (why L=1)

Aspect ratio 4:3.

> Draw a diagram explaining a light cone on a graph.
>
> Left panel, titled `L = 1  (reach radius 2)`: a sparse 16-node graph. Pick one node,
> color it blue, and label it `u`. Shade a translucent blue disc covering exactly the nodes
> within graph distance 2 of `u` — label that region `light cone of u`. Nodes inside are
> blue with a small check mark; nodes outside are gray with a small `✕`. Add a callout box
> pointing at the shaded region: `d(u,v) ≤ 2L  →  Cov(x_u, x_v | c) can be nonzero`. Add a
> second callout pointing outside: `d(u,v) > 2L  →  Cov(x_u, x_v | c) = 0  exactly`.
>
> Right panel, titled `L = 3  (reach radius 6)`: the same graph, but now the shaded disc
> covers *every* node. Big red text under it: `all 120 pairs reachable — topology imposes
> nothing. CDG ≡ permuted graph.`
>
> Between the two panels, a vertical divider. Under the whole figure, one line of caption:
> `Shallow is not a compromise. It is the only regime in which the topology carries
> information.`
>
> [+ shared style directive]

---

## Figure 3 — Aligned vs. permuted (the central claim)

Aspect ratio 16:9. This is the figure that answers "why *Clinical*".

> Draw two side-by-side panels showing the SAME graph structure with DIFFERENT node labels.
>
> Both panels show an identical 16-node graph — identical shape, identical edges, identical
> degree sequence, identical triangles. Only the labels on the nodes differ.
>
> Left panel, titled `Aligned (CDG)`, green accent: label the nodes with clinical variable
> names so that clinically related ones sit next to each other — put `creatinine`, `BUN`,
> `potassium` together in one triangle; `sodium`, `chloride`, `bicarbonate` together in
> another; `hemoglobin`, `platelets`, `WBC` in another; `SBP`, `DBP`, `heart rate` in
> another. Draw a thick green line between `creatinine` and `BUN` and label it
> `|ρ| = 0.66 — inside the light cone ✓`.
>
> Right panel, titled `Permuted (isomorphic)`, red accent: the same graph, but the clinical
> labels are scrambled so that `creatinine` and `BUN` land far apart. Draw a thick red dashed
> line between them, routed the long way around the graph, and label it
> `|ρ| = 0.66 — distance 5, outside the light cone ✗ unrepresentable`.
>
> Under both panels, a shared caption bar reading: `Same nodes. Same edges. Same degree
> sequence. Same triangles. The ONLY difference is which clinical pair sits where.`
>
> [+ shared style directive]

**What must be exactly right:** the two graphs must be visually *identical in shape*. If the
reader can tell them apart by their structure, the figure has destroyed the whole point.

---

## Figure 4 — Why classical parameters cannot fake it (optional, nice-to-have)

Aspect ratio 1:1, small. Use this one only if the README needs it; Figure 1 stage 4 already
carries most of the message.

> Draw a simple contrast diagram, two rows.
>
> Top row, labeled `Our decoder` and marked with a green check: 16 values `q_1 ... q_16`
> entering 16 *separate*, unconnected boxes `h_1 ... h_16`, each producing one output. No
> lines cross between boxes. Caption: `∂x̃_u / ∂q_v = 0 for u ≠ v (Jacobian is exactly
> diagonal). Classical parameters cannot create cross-feature dependence.`
>
> Bottom row, labeled `A dense decoder` and marked with a red ✕: the same 16 values entering
> one big fully-connected MLP block, with many crossing lines, producing 16 outputs.
> Caption: `Dependence could come from anywhere. The quantum core proves nothing.`
>
> [+ shared style directive]

---

## Figure 5 — Where the data comes from (cohort construction)

Aspect ratio 16:9. This answers "what exactly did you extract, and from which tables".

> Draw a data-provenance diagram, left to right, in three columns.
>
> **Left column — "MIMIC-IV v3.1"** (dark blue header bar). Five stacked table icons, each
> labeled with its module and table name:
> `hosp / patients`, `hosp / admissions`, `icu / icustays`, `icu / chartevents`,
> `hosp / labevents`. Under the column, small gray text: `PhysioNet credentialed access ·
> Data Use Agreement`.
>
> **Middle column — "24-hour landmark cohort"**. Draw a horizontal timeline for one ICU stay.
> Mark `ICU admission` at t=0. Shade the interval from t=0 to t=24h in orange and label it
> `observation window — features are summarized here`. Draw a vertical dashed line at t=24h
> labeled `landmark`. Shade everything after the landmark in light gray and label it
> `outcome window — mortality after 24h`. Add a note under the timeline:
> `stays shorter than 24h are excluded — no outcome could be observed`.
>
> **Right column — "Model inputs"**. Two stacked boxes.
> Upper box, orange, titled `16 generated features`: two sub-groups —
> `6 vital signs (mean over 24h)` listing `heart rate, SBP, DBP, respiratory rate, SpO2,
> temperature`, and `10 labs (median over 24h)` listing `WBC, glucose, sodium, potassium,
> chloride, bicarbonate, creatinine, BUN, hemoglobin, platelets`.
> Lower box, gray, titled `condition vector c`: `mortality after 24h`, `age`, `sex`,
> `ICU type`.
>
> Draw arrows from `icu / chartevents` into the vital signs group, from `hosp / labevents`
> into the labs group, and from `patients / admissions / icustays` into the condition vector
> box.
>
> Bottom-right, a small separate box outlined in a dashed red line, labeled
> `MAP — extracted but NEVER generated (evaluation only)` with a tiny footnote arrow pointing
> to Figure 6.
>
> [+ shared style directive]

**What must be exactly right:** the landmark. Features come from *before* t=24h, the outcome
from *after*. If the figure blurs that, it is drawing target leakage.

---

## Figure 6 — The MAP trap (why one feature was removed)

Aspect ratio 4:3. Short and punchy. This is the most memorable finding in the whole
data-preparation stage and it is worth its own figure.

> Draw a two-panel "before / after" diagram about removing a variable.
>
> Left panel, titled `With MAP` and outlined in red. Three nodes: `SBP`, `DBP`, `MAP`,
> arranged as a triangle with `MAP` at the top. Draw solid arrows from `SBP` to `MAP` and
> from `DBP` to `MAP`, and label the pair with a formula box:
> `MAP ≈ (SBP + 2·DBP) / 3     R² = 0.860`. Then draw the `SBP`—`DBP` edge as a thick RED
> line labeled `ρ = −0.508`, with a red warning callout: `physiologically wrong — a collider
> artifact. Conditioning on a deterministic function of SBP and DBP flips the sign.`
>
> Right panel, titled `MAP removed` and outlined in green. Only two nodes, `SBP` and `DBP`,
> joined by a thick GREEN edge labeled `ρ = +0.499`, with a green callout: `the real
> physiological relationship`. Beside it, a small gray box: `WBC added in its place — the
> inflammation axis was missing entirely`.
>
> Under both panels, one caption bar: `An arithmetic identity is not a clinical dependency.
> Leaving MAP in would have put a physiologically incorrect edge into the CDG — and let a
> reviewer reduce our result to "you recovered division by three."`
>
> [+ shared style directive]

---

---

# Revision prompts (round 2)

Paste each of these together with the shared style directive.

## Fix for Figure 1 — pipeline (round 3)

Round 2 fixed the local heads (`h_1` … `h_16`) and the critic loop — keep both. **The CDG in
Stage 2 is still wrong.** Three defects:

1. **Every cluster is drawn as a 4-cycle (a square). There are no triangles anywhere.** The
   triangles are the mechanism the whole paper rests on.
2. **The three bridges are missing.** The four clusters float unconnected; the graph must be
   connected end to end.
3. **Node `15` appears twice and `16` is missing.**

> Keep the five-stage layout, the colors, the `h_1(q_1,c)` … `h_16(q_16,c)` heads, the critic
> loop, and the dotted guidelines between the CDG and the RZZ links. Redraw **only the graph in
> Stage 2**, and re-derive the RZZ links in Stage 3 from it.
>
> Draw the reference graph exactly as specified in "The reference graph" above: 16 nodes, 19
> edges, four clusters, **each cluster is a triangle with one extra node hanging off it** — not
> a square — and the four clusters are chained together by three bridges so the whole graph is
> connected.
>
> Label the graph nodes with the clinical names (`heart rate`, `SBP`, `DBP`, `respiratory
> rate`, `SpO2`, `temperature`, `WBC`, `glucose`, `sodium`, `potassium`, `chloride`,
> `bicarbonate`, `creatinine`, `BUN`, `hemoglobin`, `platelets`), each name appearing exactly
> once, with a small `q_1` … `q_16` tag underneath so the reader can follow a name to its qubit
> wire.
>
> The vertical RZZ links in Stage 3 must connect exactly the 19 pairs listed in the edge list —
> including the three bridge links, which will span between distant wires.
>
> Before exporting, check: 4 visible triangles · 3 bridges · connected graph · 16 distinct names
> · no name repeated · no node with more than 3 edges.

## Fix for Figure 3 — aligned vs permuted (redraw)

> Discard the ring layout entirely — a ring has no triangles, yet the caption claims the two
> graphs share triangles, and a ring is separately one of our control conditions, so drawing
> it here misleads.
>
> Draw two side-by-side panels showing the SAME 16-node graph with DIFFERENT node labels.
>
> The graph (identical in both panels): four clusters of four, each cluster containing a
> triangle, joined by three bridges. Maximum degree 3. 19 edges. Lay the two panels out with
> *pixel-identical geometry* — same node positions, same edges, same shape. Only the text on
> the nodes differs.
>
> Left panel, titled `Aligned (CDG)`, green accent. Label the nodes so that clinically related
> variables land in the same triangle:
> `heart rate, SBP, DBP` + `respiratory rate` · `SpO2, temperature, WBC` + `glucose` ·
> `sodium, potassium, chloride` + `bicarbonate` · `creatinine, BUN, hemoglobin` + `platelets`.
> Draw the `creatinine`—`BUN` edge as a thick green line and label it
> `|ρ| = 0.66 — adjacent, inside the light cone ✓`.
>
> Right panel, titled `Permuted (isomorphic)`, red accent. Same graph, but the 16 clinical
> labels are scrambled across the nodes so that `creatinine` and `BUN` end up in *different*
> clusters, far apart. Draw a thick red dashed line between them, routed the long way through
> the graph, and label it `|ρ| = 0.66 — distance 5, outside the light cone ✗ unrepresentable`.
>
> Shared caption bar underneath: `Same nodes. Same edges. Same degree sequence. Same
> triangles. The ONLY difference is which clinical pair sits where.`
>
> Use only these 16 names — there is no pH: heart rate, SBP, DBP, respiratory rate, SpO2,
> temperature, WBC, glucose, sodium, potassium, chloride, bicarbonate, creatinine, BUN,
> hemoglobin, platelets.

## Fix for Figure 5 — data provenance

> Keep the layout, the three column headers, the timeline, and all the text. Fix the wiring
> on the right-hand side only. Currently the arrow leaving `hosp / labevents` points at the
> condition vector box, which is wrong.
>
> Correct wiring:
> - `icu / chartevents` → the `6 vital signs` box
> - `hosp / labevents` → the `10 labs` box
> - `hosp / patients`, `hosp / admissions`, `icu / icustays` → the `condition vector c` box
>
> Draw those as three clearly separated arrow paths so a reader can trace each source to its
> destination.

## Optional improvement to Figure 2 — light cone

> The current version draws the graph as a square lattice. Redraw it on the same reference
> graph used in Figures 1 and 3 (four clusters of four, each with a triangle, joined by three
> bridges), so that all three figures show the same object. Keep everything else.
>
> Left panel `L = 1 (reach radius 2)`: pick `creatinine` as the node `u`. Shade the nodes
> within graph distance 2 of it. Keep both callouts.
> Right panel `L = 3 (reach radius 6)`: the shaded region now covers every node.

---

## Plotted, not generated

These carry real numbers and will be produced with matplotlib from the result files.
Do **not** ask an image model for them — a model that guesses a bar height publishes a false
number.

| Figure | Source | Shows |
|---|---|---|
| Light-cone cliff | `RESULTS_ceiling.md` | Max reachable \|ρ\| vs graph distance; the cliff at exactly `d = 2L` |
| Alignment decay | `RESULTS_precheck.md` | Precheck `z` = +3.18 → +0.66 → 0.00 as `L` goes 1 → 2 → 3 |
| Joint ceiling | `RESULTS_ceiling_joint.md` | aligned vs permuted / distmatched / rewired / no_entangle vs floor |
| Feature table | `scripts/features.py` | The 16 features with itemid, source table, aggregation, bounds — generated by `feature_table()`, not drawn |
| CDG itself | `scripts/build_cdg.py` | The actual estimated graph on real MIMIC data. Must be the real one, never an artist's impression |

The CDG one matters: **do not let an image model draw the actual estimated clinical graph.**
Figure 1 stage 2 and Figure 3 are *schematics* and may use a stylized graph, but anywhere
the README claims "this is our CDG", it must be plotted from `build_cdg.py` output.
