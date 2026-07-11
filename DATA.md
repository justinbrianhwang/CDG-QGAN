# Data and Environment Status

Updated: 2026-07-11 (after consolidating the data archive)

## 1. Data — everything is already in hand. Nothing to download.

Both the raw data and its derivatives live in the **shared archive**.
Principle: **not a single patient-level record belongs in the code project.**

| Location | Contents | Access conditions |
|---|---|---|
| `D:\pythondata\Medical Data\Electronic Health Records\MIMIC-IV` | **MIMIC-IV v3.1** — 10GB. 364,627 patients / 546,028 admissions / 94,458 ICU stays | PhysioNet credentialed (**already held**) |
| `.../MIMIC-IV-demo-2.2` | Public demo, 100 patients. For pipeline debugging | Public (ODC-BY) |
| `.../eicu-crd` | eICU-CRD v2.0, 514MB. For external validation — outside the scope of the first paper | PhysioNet credentialed |
| `D:\pythondata\Medical Data\_derived\CDG-QGAN\` | Cohort tables, CDGs, and other **patient-level derivatives** | **Subject to the DUA. Distribution prohibited.** |
| Repo `tools/mimic-code/` | Official concept SQL. `mimic-iv/concepts/firstday/` matches our "first 24-hour summary" | Public |
| Repo `refs/` | 8 reference PDFs | Public |
| Repo `results/` | Aggregate metrics (not patient-level) | Publishable |

Paths are **managed in a single place, `scripts/paths.py`**. They can be overridden with the `MEDICAL_DATA_ROOT` environment variable.

```bash
python scripts/paths.py          # check that the paths exist
python scripts/extract_cohort.py         # v3.1 (the main experiment)
python scripts/extract_cohort.py --demo  # demo, 100 patients (debugging)
```

### Basis for identifying the MIMIC-IV version

There is no version file, so we confirmed it from row counts. `patients=364,627`, `admissions=546,028`,
and `icustays=94,458` match the official **v3.1** figures (v2.2 has roughly 300,000 patients).

### DUA warning

We placed a `.gitignore` in the archive that blocks the MIMIC/eICU paths and the data file extensions.
There is a `.git` folder, but it is not an actual repository. **If you later run `git init`, a single
`git add -A` will stage 10GB of DUA-protected patient data.** Derivatives such as the cohort and the CDG
are patient-level as well, so they fall under the same distribution ban.

### Reference verification

We looked up the arXiv IDs for [13] and [14] in the plan document, and **both are real**:
- `arXiv:2505.22533` — TabularQGAN (2025-05-28)
- `arXiv:2602.12704` — QTabGAN (2026-02-13), Kumari, Achutha, and Sivaraman — bibliographic details match

---

## 2. Cohort (v3.1, 24-hour landmark)

| Stage | Patients |
|---|---|
| First ICU stay per patient | 65,366 |
| LOS ≥ 24h (survived to the landmark) | 51,839 |
| Adults (≥ 18 years) | 51,839 |
| Excluding deaths before 24h | 51,668 |
| **y=1 (in-hospital death after 24h)** | **5,350 (10.4%)** |

This is enough to estimate a 16×16 precision matrix stably with the graphical lasso
(test 15% ≈ 7,750 patients / roughly 800 deaths).

---

## 3. Environment

### conda env: `cdg-qgan`

```bash
conda activate cdg-qgan
```

| Package | Version |
|---|---|
| Python | 3.12.13 |
| torch | 2.11.0+cu128 (CUDA confirmed working, RTX 5090 / 31.8 GiB) |
| numpy | **2.2.6 — pinned. Do not upgrade it.** |
| pennylane | 0.45.1 |
| qiskit / qiskit-aer | 2.5.0 / 0.17.2 |
| scikit-learn | 1.9.0 (GraphicalLasso) |
| pandas / scipy / networkx / statsmodels / xgboost / pyarrow | 3.0.3 / 1.18.0 / 3.6.1 / 0.14.6 / 3.3.0 / 25.0.0 |

Reproducibility: `environment.yml`, `requirements.lock.txt`

### Why numpy must be pinned at 2.2.6

Installing pennylane/qiskit naively upgrades numpy to 2.4.x, but **torch 2.11 (cu128) is built against
the numpy 2.2 ABI, so loading `shm.dll` breaks** (`OSError: [WinError 127]`).
This is exactly how the base conda environment got broken. Always apply the constraint file when adding packages:

```bash
python -m pip install -c .backup/constraints.txt <package>
```

### Miscellaneous

- The Windows console uses cp949, so Korean output is mangled. In scripts, use
  `sys.stdout.reconfigure(encoding="utf-8")` + `PYTHONIOENCODING=utf-8`.
- `conda run` cannot handle multi-line `python -c`. Invoke the env's python directly:
  `C:\Users\sunju\miniconda3\envs\cdg-qgan\python.exe`
- 2026-07-11: at the user's request, **all 123 pypi packages were removed from base conda**.
  To restore: `pip install -r .backup/base-pip-freeze-20260711.txt`
  (the removal list includes packages for other purposes, such as anthropic, google-genai, fastapi, and gymnasium)

---

## 4. Verification complete

`scripts/smoke_test_env.py` — all 3 core items from Appendix B of the plan document PASS:

- **B-1** `Z` measured immediately after `RZZ` → entanglement gradient = `-1.5e-17` (zero)
- **B-2** `RZZ` → non-commuting local `RX/RY` mixing → `Z` → gradient = `+0.46` (nonzero)
- **B-3** Our hand-written torch statevector agrees with PennyLane (`|diff| = 3.3e-08`)

**B-1/B-2 are the core justification for the v2 design**: without local mixing, the parameters of the
final entangling layer are not trained at all.

`scripts/qsim_lightcone.py` — the light-cone subcircuit agrees with the full statevector to `1e-6`.
At n=16, L=1 this is a **4096x saving** (65,536 dimensions → 16 dimensions).

---

## 5. Next

**Work order and specifications are in `HANDOFF.md`, revisions to the plan document are in `REVISIONS.md`, and results are in `RESULTS_*.md`.**

The gate: `scripts/precheck_alignment.py`. Without any training, it judges from the graph structure alone
whether the confirmatory experiment can succeed. **If it FAILs, do not burn GPU — change the design.**
