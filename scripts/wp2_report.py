"""Aggregate the WP-2 shards and run the analysis. This is the only place WP-2 is interpreted.

`wp2.py` shards deliberately do NOT analyse themselves: a shard holds a subset of the variants,
and a shard that computed "CDG − distance-matched" against controls it never ran would be
producing a number out of thin air. So each shard writes raw per-seed scores and stops. This
script collects them, checks the design is complete, applies the floor gate, and only then
computes the contrasts.

The floor gate (REVISIONS §E-2, §E-8)
-------------------------------------
If the trained CDG model does not beat the floor, NOTHING below it is interpretable. When nothing
has learned any dependency, every topology scores alike, and a bootstrap over near-identical
numbers will cheerfully report a "significant" difference of 0.0007. We nearly published exactly
that. The gate is not a formality — it has fired twice on real runs and was right both times.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from paths import RESULTS  # noqa: E402

EXPECTED = ["cdg", "permuted_0", "permuted_1", "permuted_2",
            "distmatched_0", "distmatched_1", "distmatched_2",
            "rewired", "ring", "no_entangle"]

# These are properties of the GRAPH the run used, not of WP-2. They were the Δ=3 numbers and were
# left behind when the architecture moved to Δ=4 (§E-9) — which silently made the "gap closed"
# column wrong, because it is measured against the ceiling.
CEILING = 0.0300      # RESULTS_design.md, Δ=4 @ 4000×4 — reachable with the GAN removed
BOUND = 0.0140        # Corollary 1 — no L=1 Δ=4 circuit can beat this

# The floor permutes the features AND c, so it destroys the x–c relation as well as the
# cross-feature one. That was the right null for v2. It is the WRONG null for v3, whose 1-D
# marginal term makes E[x_u | c] correct on purpose: the evaluator residualizes on a fixed basis
# (1, c, c², c_a·c_b), whatever of E[x_u | c] lies outside that span survives, and — being a
# function of c — it is shared across features, which a correlation estimator reports as
# dependence. `zr` carries the same bias, so a model that fits the conditional marginals scores
# BELOW the floor while creating no dependence at all. `null_condmarg.py` measures it: a surrogate
# with exactly zero conditional cross-feature dependence scores 0.0942, 4.3% under the floor.
#
# Report both. The floor gates the run; the honest null is what a topology claim must beat.
HONEST_NULL = 0.0942  # null_condmarg.py — zero dependency, correct conditional marginals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=1)
    args = ap.parse_args()

    files = sorted(RESULTS.glob(f"wp2_L{args.depth}_shard*.json"))
    single = RESULTS / f"wp2_L{args.depth}.json"
    if single.exists():
        files.append(single)
    if not files:
        sys.exit(f"no WP-2 result files for L={args.depth} in {RESULTS}")

    # Every shard must have been produced by the same configuration, apart from its shard index.
    # Mixing a Δ=3 shard into a Δ=4 table, or a v2-loss shard into a v3 table, would be invisible
    # in the output and would corrupt every contrast below. Refuse rather than average.
    res, floor, cfgs = {}, [], []
    for f in files:
        d = json.loads(f.read_text())
        c = dict(d.get("config") or {})
        c.pop("shard", None)
        cfgs.append((f.name, c))
        res.update(d["results"])
        floor.append(d["floor"])

    if any(not c for _, c in cfgs):
        sys.exit("a result file has no `config` block — it predates the config check.\n"
                 "  Delete it and re-run that shard. Refusing to guess what produced it.")
    base = cfgs[0][1]
    for name, c in cfgs[1:]:
        if c != base:
            diff = {k: (base.get(k), c.get(k)) for k in set(base) | set(c)
                    if base.get(k) != c.get(k)}
            sys.exit(f"shards disagree on their configuration — refusing to aggregate.\n"
                     f"  {cfgs[0][0]} vs {name}: {diff}")

    floor = float(np.mean(floor))
    steps, batch, seeds = base["steps"], base["batch"], base["seeds"]

    missing = [v for v in EXPECTED if v not in res]
    if missing:
        sys.exit(f"design incomplete — missing variants: {', '.join(missing)}\n"
                 f"  have: {', '.join(sorted(res))}\n"
                 f"  Refusing to analyse a partial design.")

    print("=" * 94)
    print(f"WP-2 — real MIMIC-IV · L={args.depth} · {seeds} seeds · {steps} steps · batch {batch}")
    print("=" * 94)
    print(f"  floor       {floor:.4f}   zero dependency AND no x–c relation (gates the run)")
    print(f"  honest null {HONEST_NULL:.4f}   zero dependency, correct conditional marginals "
          f"(null_condmarg.py)")
    print(f"  ceiling     {CEILING:.4f}   reachable with the GAN removed (RESULTS_design.md)")
    print(f"  bound       {BOUND:.4f}   no L=1 Δ=4 circuit can beat this (Corollary 1)")
    print()
    print(f"  {'variant':<16} {'120-pair error':>16} {'vs floor':>10} {'vs null':>9} "
          f"{'% of reachable gap closed':>27}")
    print("  " + "-" * 84)

    for name in EXPECTED:
        r = np.array(res[name])
        m, sd = r.mean(), r.std()
        # how much of the distance between "create no dependency" and "the best this circuit
        # could do" has actually been closed. Measured from the honest null, not the floor: the
        # 4% the floor hands out for free is not something the topology earned.
        closed = (HONEST_NULL - m) / (HONEST_NULL - CEILING) * 100
        print(f"  {name:<16} {m:>10.4f} ± {sd:.4f} {(m - floor) / floor * 100:>+9.1f}% "
              f"{(m - HONEST_NULL) / HONEST_NULL * 100:>+8.1f}% {closed:>25.0f}%")

    cdg = np.array(res["cdg"])

    print()
    print("=" * 94)
    if cdg.mean() >= floor:
        print("  ** STOP — the trained CDG model LOSES to a model that creates no dependency **")
        print(f"     CDG {cdg.mean():.4f}  vs  floor {floor:.4f}")
        print()
        print("     Nothing has learned any dependency, so every topology scores alike and the")
        print("     contrasts are noise. Do NOT read them as evidence for or against the")
        print("     hypothesis. Check, in order: steps (>= 8000), batch (512), the copula +")
        print("     batch-aware critic, and that the metric is eval_dep.partial_corr_c.")
        print("     See REVISIONS §E-8.")
        print("=" * 94)
        sys.exit(1)

    print(f"  gate passed: CDG {cdg.mean():.4f} < floor {floor:.4f}. The contrasts are meaningful.")
    print()

    def group(prefix):
        return np.concatenate([res[k] for k in res if k.startswith(prefix)])

    rng = np.random.default_rng(0)
    for label, ctrl in (("permuted (3 graphs)", group("permuted")),
                        ("distance-matched (3 graphs)", group("distmatched")),
                        ("rewired", np.array(res["rewired"])),
                        ("ring", np.array(res["ring"])),
                        ("no_entangle", np.array(res["no_entangle"]))):
        d = cdg.mean() - ctrl.mean()
        boot = [rng.choice(cdg, len(cdg)).mean() - rng.choice(ctrl, len(ctrl)).mean()
                for _ in range(5000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        verdict = ("CDG better" if hi < 0 else
                   "not significant" if lo < 0 < hi else "CDG WORSE")
        print(f"  CDG − {label:<28} = {d:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   {verdict}")

    print()
    print("  Primary : CDG − permuted < 0 with the CI excluding 0.")
    print("  Decisive: CDG − distance-matched < 0. Beating the isomorphic permutation alone could")
    print("            be graph combinatorics; distance matching is what isolates the clinical claim.")
    print()
    print("  Note: on real data the permutation null is a WEAK-signal control, not a zero-signal")
    print("  one. MIMIC's strong dependencies are clustered (electrolytes, renal), so a random")
    print("  relabelling of a 21-edge graph lands on a few genuine pairs by luck. That makes the")
    print("  CDG−permuted gap a conservative estimate of the effect, not an inflated one.")
    print("=" * 94)

    out = RESULTS / f"wp2_L{args.depth}_combined.json"
    out.write_text(json.dumps({"floor": floor, "ceiling": CEILING, "bound": BOUND,
                               "results": res, "seeds": seeds, "steps": steps,
                               "batch": batch}, indent=2))
    print(f"\n  saved: {out}")


if __name__ == "__main__":
    main()
