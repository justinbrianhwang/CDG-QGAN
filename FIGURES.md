# Figure prompts for the README

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
