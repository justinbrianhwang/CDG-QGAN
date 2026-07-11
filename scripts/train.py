"""CDG-QGAN 학습 (conditional WGAN-GP) + 의존성 복원 평가.

v2 §8.10: primary model은 WGAN-GP만 사용한다.
  - CDG edge loss, 상관 손실, 임상 제약 손실, auxiliary classifier 없음
  - 평가 지표를 직접 최적화하는 순환 논증을 제거하기 위함
  - 따라서 관찰되는 구조 복원 차이는 오직 회로 토폴로지의 inductive bias다

평가 [리뷰 B-2 반영]:
  primary  = 120쌍 전체 조건부 의존성 오차 (위음성 + 위양성 모두 잡음)
  보조     = held-out 간선 오차 (비순환성 확인용)
  둘 다 nonparanormal 공간에서 계산한다 [리뷰 A-2].
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).parent))

from model import CDGQGAN, Critic  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Cfg:
    depth: int = 1
    batch: int = 256
    critic_steps: int = 5
    lr_g: float = 5e-5
    lr_d: float = 1e-4
    lambda_gp: float = 10.0
    steps: int = 3000
    head_width: int = 8
    seed: int = 0
    use_cuda_graph: bool = True  # 검증용으로 끌 수 있어야 한다 (PM 요구)


# ---------------------------------------------------------------------------
# 평가: nonparanormal 공간의 조건부 부분상관
# ---------------------------------------------------------------------------
def _npn(X: np.ndarray) -> np.ndarray:
    """rank -> inverse normal. HDE는 반드시 이 공간에서 계산해야 한다 [리뷰 A-2].

    원 단위 Pearson으로 재면 폭-8 head가 못 맞추는 heavy-tail 주변분포 오차가
    의존성 지표를 오염시킨다. 단조 head에 불변인 공간에서 재야 한다.
    """
    N = X.shape[0]
    return np.column_stack([norm.ppf((rankdata(X[:, j]) - 0.5) / N) for j in range(X.shape[1])])


def partial_corr_matrix(X: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """부분상관 행렬. real/synthetic에 완전히 동일한 정규화를 적용해야 한다."""
    S = np.corrcoef(_npn(X), rowvar=False)
    P = np.linalg.inv(S + ridge * np.eye(S.shape[0]))
    d = np.sqrt(np.diag(P))
    R = -P / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


def dependency_error(X_real: np.ndarray, X_syn: np.ndarray, pairs=None) -> float:
    """Fisher-z 공간의 평균 절대 부분상관 오차."""
    Rr, Rs = partial_corr_matrix(X_real), partial_corr_matrix(X_syn)
    zr, zs = np.arctanh(np.clip(Rr, -0.999, 0.999)), np.arctanh(np.clip(Rs, -0.999, 0.999))
    p = Rr.shape[0]
    if pairs is None:  # 120쌍 전체 (primary) — 위양성도 잡는다
        pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
    return float(np.mean([abs(zs[i, j] - zr[i, j]) for i, j in pairs]))


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------
def gradient_penalty(critic, x_real, x_fake, c) -> torch.Tensor:
    a = torch.rand(x_real.size(0), 1, device=x_real.device)
    xh = (a * x_real + (1 - a) * x_fake).requires_grad_(True)
    d = critic(xh, c)
    g, = torch.autograd.grad(d.sum(), xh, create_graph=True)
    return ((g.norm(2, dim=1) - 1) ** 2).mean()


def train(X: np.ndarray, C: np.ndarray, edges, cfg: Cfg, log_every: int = 0):
    """X: (N, n) 실제 특징 (이미 scaling됨).  C: (N, cond_dim) 조건 벡터."""
    torch.manual_seed(cfg.seed)
    N, n = X.shape
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    Ct = torch.tensor(C, dtype=torch.float32, device=DEVICE)

    G = CDGQGAN(n, edges, cfg.depth, C.shape[1], cfg.head_width, seed=cfg.seed).to(DEVICE)
    D = Critic(n, C.shape[1]).to(DEVICE)
    capturable = DEVICE == "cuda" and cfg.use_cuda_graph
    og = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(0.0, 0.9), capturable=capturable)
    od = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(0.0, 0.9), capturable=capturable)

    def sample_real(b):
        i = torch.randint(0, N, (b,), device=DEVICE)
        return Xt[i], Ct[i]

    def sample_z(b):
        return 2 * torch.rand(b, n, device=DEVICE) - 1  # local latent, iid

    critic_graph = generator_graph = None
    if DEVICE == "cuda" and cfg.use_cuda_graph:
        static_xr = torch.empty(cfg.batch, n, device=DEVICE)
        static_c = torch.empty(cfg.batch, C.shape[1], device=DEVICE)
        static_z = torch.empty(cfg.batch, n, device=DEVICE)

        def critic_body():
            with torch.no_grad():
                xf = G(static_z, static_c)
            loss = (D(xf, static_c).mean() - D(static_xr, static_c).mean()
                    + cfg.lambda_gp * gradient_penalty(D, static_xr, xf, static_c))
            od.zero_grad(set_to_none=True)
            loss.backward()
            od.step()
            return loss

        def generator_body():
            loss = -D(G(static_z, static_c), static_c).mean()
            og.zero_grad(set_to_none=True)
            loss.backward()
            og.step()
            return loss

        # CUDA Graph capture needs optimizer state and allocator pools warmed up.
        # Restore parameters and optimizer state afterwards so warmup is invisible.
        saved = [p.detach().clone() for p in list(G.parameters()) + list(D.parameters())]
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                critic_body()
                generator_body()
        torch.cuda.current_stream().wait_stream(warm)

        critic_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(critic_graph):
            ld = critic_body()
        generator_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(generator_graph):
            lg = generator_body()

        with torch.no_grad():
            for p, value in zip(list(G.parameters()) + list(D.parameters()), saved):
                p.copy_(value)
            for opt in (og, od):
                for state in opt.state.values():
                    for value in state.values():
                        if torch.is_tensor(value):
                            value.zero_()

    for step in range(cfg.steps):
        for _ in range(cfg.critic_steps):
            xr, c = sample_real(cfg.batch)
            if critic_graph is not None:
                static_xr.copy_(xr)
                static_c.copy_(c)
                static_z.copy_(sample_z(cfg.batch))
                critic_graph.replay()
            else:
                with torch.no_grad():
                    xf = G(sample_z(cfg.batch), c)
                ld = D(xf, c).mean() - D(xr, c).mean() + cfg.lambda_gp * gradient_penalty(D, xr, xf, c)
                od.zero_grad(set_to_none=True)
                ld.backward()
                od.step()

        _, c = sample_real(cfg.batch)
        if generator_graph is not None:
            static_c.copy_(c)
            static_z.copy_(sample_z(cfg.batch))
            generator_graph.replay()
        else:
            lg = -D(G(sample_z(cfg.batch), c), c).mean()
            og.zero_grad(set_to_none=True)
            lg.backward()
            og.step()

        if log_every and step % log_every == 0:
            print(f"    step {step:5d}  L_D={ld.item():+.4f}  L_G={lg.item():+.4f}", flush=True)

    return G, D


@torch.no_grad()
def generate(G: CDGQGAN, C: np.ndarray, n_samples: int, seed: int = 0) -> np.ndarray:
    """조건 분포를 실제와 맞춰 합성 표본 생성."""
    torch.manual_seed(seed)
    idx = np.random.default_rng(seed).integers(0, len(C), n_samples)
    c = torch.tensor(C[idx], dtype=torch.float32, device=DEVICE)
    z = 2 * torch.rand(n_samples, G.n, device=DEVICE) - 1
    return G(z, c).cpu().numpy()
