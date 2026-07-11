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

CEILING = 0.0437      # RESULTS_ceiling_real.md — reachable without a GAN
BOUND = 0.0331        # Corollary 1 — no L=1 CDG circuit can beat this


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

    res, floor, steps, batch, seeds = {}, [], None, None, None
    for f in files:
        d = json.loads(f.read_text())
        res.update(d["results"])
        floor.append(d["floor"])
        steps, batch, seeds = d["steps"], d.get("batch"), d["seeds"]
    floor = float(np.mean(floor))

    missing = [v for v in EXPECTED if v not in res]
    if missing:
        sys.exit(f"design incomplete — missing variants: {', '.join(missing)}\n"
                 f"  have: {', '.join(sorted(res))}\n"
                 f"  Refusing to analyse a partial design.")

    print("=" * 94)
    print(f"WP-2 — real MIMIC-IV · L={args.depth} · {seeds} seeds · {steps} steps · batch {batch}")
    print("=" * 94)
    print(f"  floor   {floor:.4f}   a model that creates no dependency at all")
    print(f"  ceiling {CEILING:.4f}   reachable with the GAN removed (RESULTS_ceiling_real.md)")
    print(f"  bound   {BOUND:.4f}   no L=1 CDG circuit can beat this (Corollary 1)")
    print()
    print(f"  {'variant':<16} {'120-pair error':>16} {'vs floor':>10} "
          f"{'% of reachable gap closed':>27}")
    print("  " + "-" * 74)

    for name in EXPECTED:
        r = np.array(res[name])
        m, sd = r.mean(), r.std()
        # how much of the distance between "do nothing" and "the best this circuit could do"
        # has actually been closed
        closed = (floor - m) / (floor - CEILING) * 100
        print(f"  {name:<16} {m:>10.4f} ± {sd:.4f} {(m - floor) / floor * 100:>+9.1f}% "
              f"{closed:>25.0f}%")

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
