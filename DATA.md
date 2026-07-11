# 데이터 및 환경 현황

갱신: 2026-07-11 (데이터 아카이브 통합 후)

## 1. 데이터 — 전부 확보 완료. 받을 것 없음.

원본도 파생물도 **공용 아카이브**에 둔다.
원칙: **코드 프로젝트에는 환자 단위 데이터가 한 톨도 없다.**

| 위치 | 내용 | 접근 조건 |
|---|---|---|
| `D:\pythondata\Medical Data\Electronic Health Records\MIMIC-IV` | **MIMIC-IV v3.1** — 10GB. 환자 364,627 / 입원 546,028 / ICU 입실 94,458 | PhysioNet credentialed (**이미 보유**) |
| `.../MIMIC-IV-demo-2.2` | 공개 demo, 환자 100명. 파이프라인 디버그용 | 공개 (ODC-BY) |
| `.../eicu-crd` | eICU-CRD v2.0, 514MB. 외부 검증용 — 첫 논문 범위 밖 | PhysioNet credentialed |
| `D:\pythondata\Medical Data\_derived\CDG-QGAN\` | 코호트 테이블, CDG 등 **환자 단위 파생물** | **DUA 대상. 배포 금지** |
| 리포 `tools/mimic-code/` | 공식 concept SQL. `mimic-iv/concepts/firstday/` 가 "첫 24시간 요약"과 일치 | 공개 |
| 리포 `refs/` | 참고문헌 PDF 8편 | 공개 |
| 리포 `results/` | 집계 지표 (환자 단위 아님) | 공개 가능 |

경로는 **`scripts/paths.py`에서 단일 관리**한다. 환경변수 `MEDICAL_DATA_ROOT`로 덮어쓸 수 있다.

```bash
python scripts/paths.py          # 경로 존재 확인
python scripts/extract_cohort.py         # v3.1 (본 실험)
python scripts/extract_cohort.py --demo  # demo 100명 (디버그)
```

### MIMIC-IV 버전 판별 근거

버전 파일이 없어 행 수로 확인했다. `patients=364,627`, `admissions=546,028`,
`icustays=94,458` 는 **v3.1** 공식 수치와 일치한다 (v2.2는 환자 약 30만).

### DUA 주의

아카이브에 `.gitignore`를 넣어 MIMIC/eICU 경로와 데이터 확장자를 차단해뒀다.
`.git` 폴더가 있으나 실제 저장소는 아니다. **나중에 `git init` 하면 `git add -A`
한 번에 10GB의 DUA 보호 환자 데이터가 스테이징된다.** 코호트·CDG 같은 파생물도
환자 단위이므로 동일하게 배포 금지 대상이다.

### 참고문헌 검증

계획서 [13], [14]의 arXiv ID를 실제 조회했고 **둘 다 실재한다**:
- `arXiv:2505.22533` — TabularQGAN (2025-05-28)
- `arXiv:2602.12704` — QTabGAN (2026-02-13), Kumari·Achutha·Sivaraman — 서지사항 일치

---

## 2. 코호트 (v3.1, 24시간 landmark)

| 단계 | 환자 수 |
|---|---|
| 환자별 최초 ICU 입실 | 65,366 |
| LOS ≥ 24h (landmark 생존) | 51,839 |
| 성인 (≥18세) | 51,839 |
| 24h 이전 사망 제외 | 51,668 |
| **y=1 (24h 이후 병원내 사망)** | **5,350 (10.4%)** |

16×16 정밀도 행렬을 graphical lasso로 안정 추정하기에 충분하다
(test 15% ≈ 7,750명 / 사망 약 800명).

---

## 3. 환경

### conda env: `cdg-qgan`

```bash
conda activate cdg-qgan
```

| 패키지 | 버전 |
|---|---|
| Python | 3.12.13 |
| torch | 2.11.0+cu128 (CUDA 동작 확인, RTX 5090 / 31.8 GiB) |
| numpy | **2.2.6 — 고정. 올리지 마십시오.** |
| pennylane | 0.45.1 |
| qiskit / qiskit-aer | 2.5.0 / 0.17.2 |
| scikit-learn | 1.9.0 (GraphicalLasso) |
| pandas / scipy / networkx / statsmodels / xgboost / pyarrow | 3.0.3 / 1.18.0 / 3.6.1 / 0.14.6 / 3.3.0 / 25.0.0 |

재현: `environment.yml`, `requirements.lock.txt`

### numpy 2.2.6을 고정해야 하는 이유

pennylane/qiskit을 그냥 설치하면 numpy를 2.4.x로 올리는데, **torch 2.11(cu128)이
numpy 2.2 ABI로 빌드되어 있어 `shm.dll` 로드가 깨진다** (`OSError: [WinError 127]`).
실제로 base conda가 이렇게 망가졌다. 패키지 추가 시 항상 constraint를 걸 것:

```bash
python -m pip install -c .backup/constraints.txt <package>
```

### 기타

- Windows 콘솔이 cp949라 한글 출력이 깨진다. 스크립트에
  `sys.stdout.reconfigure(encoding="utf-8")` + `PYTHONIOENCODING=utf-8`.
- `conda run`은 여러 줄 `python -c`를 못 다룬다. env python을 직접 호출할 것:
  `C:\Users\sunju\miniconda3\envs\cdg-qgan\python.exe`
- 2026-07-11: 사용자 요청으로 **base conda의 pypi 패키지 123개 전부 제거**.
  복구: `pip install -r .backup/base-pip-freeze-20260711.txt`
  (제거 목록에 anthropic·google-genai·fastapi·gymnasium 등 타 용도 패키지 포함)

---

## 4. 검증 완료

`scripts/smoke_test_env.py` — 계획서 부록 B 핵심 3항목 전부 PASS:

- **B-1** `RZZ` 직후 `Z` 측정 → 얽힘 gradient = `-1.5e-17` (0)
- **B-2** `RZZ` → 비가환 local `RX/RY` mixing → `Z` → gradient = `+0.46` (비영)
- **B-3** 직접 구현한 torch statevector와 PennyLane 일치 (`|diff| = 3.3e-08`)

**B-1/B-2가 v2 설계의 핵심 근거다**: local mixing을 빼면 마지막 얽힘층의 파라미터가
아예 학습되지 않는다.

`scripts/qsim_lightcone.py` — light-cone 부분회로가 전체 상태벡터와 `1e-6` 일치.
n=16, L=1에서 **4096배 절감** (65,536 차원 → 16 차원).

---

## 5. 다음

**작업 순서와 명세는 `HANDOFF.md`, 계획서 개정은 `REVISIONS.md`, 결과는 `RESULTS_*.md`.**

관문: `scripts/precheck_alignment.py`. 훈련 없이 그래프 구조만으로 확증 실험의
성공 가능성을 판정한다. **FAIL이면 GPU를 태우지 말고 설계를 바꿔야 한다.**
