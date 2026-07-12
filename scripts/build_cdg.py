"""Clinical Dependency Graph estimation (CDG-QGAN v2 §7) + fixes from the review.

Procedure:
  1. nonparanormal transform (rank -> inverse normal)
  2. residualize on a nonlinear basis of the condition vector c = (y, age, sex, icu_type)
  3. graphical lasso -> sparse precision matrix -> partial correlations
  4. bootstrap stability selection
  5. E_holdout selected first (stratified by strength, connectivity preserved), then
     E_fit built from what is left by degree-constrained Kruskal

Changes from the v2 plan, following the review:
  [review A-1] The condition vector is widened from y alone to c=(y,age,sex,icu_type).
        v2 residualized the CDG on (y,age,sex,ICU) while the generator was conditioned on
        y alone, so rho_real and rho_syn were different estimators — a bug.
  [review A-4 / E-3] **The graph and the metric must be estimated in the same space.**
        This script used to residualize in RAW units and then apply the nonparanormal
        transform. `eval_dep.partial_corr_c` — the estimator the evaluation actually uses —
        does the opposite: nonparanormal first, then residualize on a *nonlinear* basis of c.
        Those are different estimators, so the graph we drew and the quantity we measured were
        not the same object. Both now use `eval_dep`.
        Order matters: subtracting a linear fit in raw units breaks the monotone relation
        between the residual and the qubit observable, which is the whole reason the metric
        lives in nonparanormal space (it must be invariant to the monotone local heads).
        The basis matters too: c enters the circuit through RY/RZ angles, i.e. nonlinearly,
        and a linear design matrix leaves half of its effect behind (measured: the spurious
        non-edge dependency only fell from 0.0926 to 0.0532).
  [review C-3] Taking E_fit as "70% of E_candidate" yields fewer than the 15 edges needed
        to connect 16 nodes, so the graph falls apart and filler edges dilute the topology.
  [review B]   Report the distribution of graph distances over the held-out pairs. We need
        to know up front whether Delta_HDE is explained by nothing more than how those
        distances happen to fall.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.covariance import graphical_lasso

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from eval_dep import cond_basis, npn  # noqa: E402
from features import CONDITION_VARS, CORE16  # noqa: E402

MAX_DEGREE = 3       # v2 §7.6
STABILITY_TAU = 0.70  # v2 §7.5
N_BOOTSTRAP = 100


def npn_residualize(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """nonparanormal, then residualize on a nonlinear basis of c.

    **The same transform `eval_dep.partial_corr_c` applies to both the real and the synthetic
    data.** The graph and the metric have to be defined in the same space, or we are drawing one
    object and measuring another [review A-4, E-3].
    """
    Xn = npn(X)
    D = cond_basis(C)
    beta, *_ = np.linalg.lstsq(D, Xn, rcond=None)
    return Xn - D @ beta


def partial_corr(X: np.ndarray, alpha: float) -> np.ndarray:
    """graphical lasso -> partial correlation matrix (v2 §7.4)."""
    S = np.corrcoef(X, rowvar=False)
    _, prec = graphical_lasso(S, alpha=alpha, max_iter=200)
    d = np.sqrt(np.diag(prec))
    rho = -prec / np.outer(d, d)
    np.fill_diagonal(rho, 1.0)
    return rho


def stability(X: np.ndarray, alpha: float, B: int, rng: np.random.Generator) -> np.ndarray:
    """Frequency with which each edge is selected across the subsampling repetitions (v2 §7.5)."""
    n, p = X.shape
    m = int(0.8 * n)
    cnt = np.zeros((p, p))
    for _ in range(B):
        idx = rng.choice(n, m, replace=False)
        try:
            rho = partial_corr(X[idx], alpha)
        except Exception:
            continue
        cnt += (np.abs(rho) > 1e-4).astype(float)
    np.fill_diagonal(cnt, 0)
    return cnt / B


def build_graph(rho, stab, names, rng, n_holdout_strong=3):
    """Select E_holdout first, stratified by edge strength, then build E_fit = MST ∪ top-r from
    what is left.

    The order matters. If E_fit is built first as MST ∪ top-r, it takes all the strong edges,
    and E_holdout is left with nothing but weak relationships of |rho|~0.05 -> the HDE would
    then be measuring nothing but estimation noise. (Confirmed in an actual demo run: every
    held-out edge had |rho| <= 0.16.)

    Hence:
      1) hold out n_holdout_strong of the strong edges, choosing only those whose removal
         still leaves the rest connected
      2) hold out some weak edges as well (to cover the strength spectrum)
      3) build E_fit = MST ∪ top-r from the remaining edges -> connectivity guaranteed, no
         filler edges needed
    """
    p = len(names)
    w = np.abs(rho) * (stab >= STABILITY_TAU)
    np.fill_diagonal(w, 0)
    cand = sorted(((i, j, w[i, j]) for i in range(p) for j in range(i + 1, p) if w[i, j] > 0),
                  key=lambda e: -e[2])

    def connected_without(excluded: set) -> bool:
        H = nx.Graph()
        H.add_nodes_from(range(p))
        H.add_edges_from((i, j) for i, j, _ in cand if tuple(sorted((i, j))) not in excluded)
        return nx.is_connected(H)

    # 1) hold out the strong edges (subject to keeping the graph connected)
    holdout: set = set()
    for i, j, _ in cand:
        if len(holdout) >= n_holdout_strong:
            break
        e = tuple(sorted((i, j)))
        if connected_without(holdout | {e}):
            holdout.add(e)

    # 2) hold out some weak edges as well (to cover the strength spectrum)
    weak = [tuple(sorted((i, j))) for i, j, _ in cand[len(cand) // 2:]
            if tuple(sorted((i, j))) not in holdout]
    rng.shuffle(weak)
    for e in weak[: max(3, len(weak) // 3)]:
        if connected_without(holdout | {e}):
            holdout.add(e)

    # 3) build E_fit from the remaining edges — enforce the degree constraint Delta<=3 from
    #    the start.
    #
    #    nx.maximum_spanning_tree does not constrain the degree, so it yields a dense graph
    #    with maxdeg=5. The diameter then shrinks, graph distance loses its discriminative
    #    power, and the light-cone-based confirmatory test is neutralized (on the demo, at
    #    L=2 the CDG and the permuted graph were identical at 12/13).
    #    -> run Kruskal ourselves under the degree constraint (degree-constrained spanning
    #    forest).
    rest = [(i, j, wt) for i, j, wt in cand if tuple(sorted((i, j))) not in holdout]

    G = nx.Graph()
    G.add_nodes_from(range(p))
    uf = {i: i for i in range(p)}

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    # 3a) Kruskal in weight order + degree constraint -> spanning forest
    for i, j, _ in rest:
        if G.degree(i) < MAX_DEGREE and G.degree(j) < MAX_DEGREE and find(i) != find(j):
            G.add_edge(i, j)
            uf[find(i)] = find(j)

    # 3b) if components are still disconnected, join them at the highest weight available
    #     between nodes that still have degree headroom
    while not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        best = None
        for a in range(len(comps)):
            for b in range(a + 1, len(comps)):
                for u in comps[a]:
                    for v in comps[b]:
                        if G.degree(u) < MAX_DEGREE and G.degree(v) < MAX_DEGREE:
                            if best is None or w[u, v] > best[2]:
                                best = (u, v, w[u, v])
        if best is None:  # no way to connect while respecting the degree constraint
            raise RuntimeError(f"Cannot connect under Delta<={MAX_DEGREE}. Raise MAX_DEGREE.")
        G.add_edge(best[0], best[1])

    # 3c) add the highest-weight edges that still fit in the degree headroom (top-r)
    for i, j, _ in rest:
        if not G.has_edge(i, j) and G.degree(i) < MAX_DEGREE and G.degree(j) < MAX_DEGREE:
            G.add_edge(i, j)

    E_fit = [tuple(sorted(e)) for e in G.edges()]
    return G, E_fit, sorted(holdout)


def report_holdout_distances(G: nx.Graph, E_holdout: list, names: list[str], rho: np.ndarray) -> None:
    """[review B] Distribution of graph distances over the held-out pairs. Check up front whether
    Delta_HDE is explained by distance alone."""
    print("\n  CDG graph distance of the held-out pairs (this decides light cone reachability)")
    dists = []
    for (i, j) in E_holdout:
        d = nx.shortest_path_length(G, i, j) if nx.has_path(G, i, j) else 99
        dists.append(d)
        print(f"    d={d}  |rho|={abs(rho[i,j]):.3f}   {names[i]} -- {names[j]}")
    if dists:
        arr = np.array(dists)
        print(f"\n    distance distribution: " + ", ".join(f"d={d}:{int((arr==d).sum())}" for d in sorted(set(dists))))
        for L in (1, 2):
            n_in = int((arr <= 2 * L).sum())
            print(f"    L={L} (reach radius {2*L}): {n_in}/{len(arr)} pairs are expressible")


def main(path: Path, alpha: float, out_name: str = "cdg.npz") -> None:
    df = pd.read_parquet(path)
    names = [f.name for f in CORE16]
    df = df.dropna(subset=names + CONDITION_VARS[1:])
    print("=" * 70)
    print(f"CDG estimation: {path}   (n={len(df):,}, p={len(names)})")
    print("=" * 70)

    if len(df) < 200:
        print("\n  ** WARNING ** the sample is far too small (demo). graphical lasso is unstable.")
        print("     This run is only for validating the pipeline; the resulting graph must not")
        print("     be used in the experiment.")

    X = df[names].to_numpy(float)
    C = df[["y", "age", "sex", "icu_type"]].to_numpy(float)  # condition vector c [review A-1]

    X = npn_residualize(X, C)   # same space as eval_dep.partial_corr_c [review A-4, E-3]
    rho = partial_corr(X, alpha)

    rng = np.random.default_rng(20260711)
    stab = stability(X, alpha, N_BOOTSTRAP, rng)

    G, E_fit, E_hold = build_graph(rho, stab, names, rng)
    print(f"\n  E_fit     : {len(E_fit)} edges  (connected={nx.is_connected(G)}, max degree={max(dict(G.degree()).values())})")
    print(f"  E_holdout : {len(E_hold)} edges  (must not be used for topology/loss/checkpointing)")

    print("\n  E_fit edges (the RZZ positions in the circuit)")
    for (i, j) in sorted(E_fit, key=lambda e: -abs(rho[e[0], e[1]])):
        print(f"    |rho|={abs(rho[i,j]):.3f}  stab={stab[i,j]:.2f}   {names[i]} -- {names[j]}")

    report_holdout_distances(G, E_hold, names, rho)

    import paths

    out = paths.PROCESSED / out_name
    np.savez(out, rho=rho, stab=stab, E_fit=np.array(E_fit), E_holdout=np.array(E_hold),
             names=np.array(names), max_degree=MAX_DEGREE)
    print(f"\n  saved: {out}   (max degree {MAX_DEGREE})")


if __name__ == "__main__":
    import paths

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.05)
    # v2 §7.6 capped the degree at 3. That cap, not the data, is what forces 72 of the 120 pairs
    # outside the L=1 light cone and puts a floor of 0.0331 under the model — 5x worse than a
    # Gaussian copula. Raising it lets the degree-constrained Kruskal keep adding edges in weight
    # order (the next strongest partial correlations), which lowers the bound while keeping L=1,
    # and with it Corollary 1 and Proposition D-2. See `scripts/design_sweep.py`.
    ap.add_argument("--max-degree", type=int, default=3)
    ap.add_argument("--out", type=str, default=None,
                    help="output filename under PROCESSED (default: cdg.npz, or cdg_d<Δ>.npz)")
    a = ap.parse_args()
    data = a.data or paths.PROCESSED / ("cohort_demo.parquet" if a.demo else "cohort_v31.parquet")
    MAX_DEGREE = a.max_degree
    out_name = a.out or ("cdg.npz" if a.max_degree == 3 else f"cdg_d{a.max_degree}.npz")
    main(data, a.alpha, out_name)
