"""환경 스모크 테스트 + 계획서 부록 B의 핵심 회로 검증.

검증 항목:
  B-1. RZZ 바로 뒤에 Z 측정 -> 얽힘 gradient가 0 (RZZ가 Z와 가환)
  B-2. RZZ 뒤에 RX/RY mixing -> gradient가 비영
  B-3. PennyLane과 torch statevector 구현이 일치
"""

import sys

import numpy as np
import pennylane as qml
import torch

sys.stdout.reconfigure(encoding="utf-8")


def scalar_grad(qnode, x):
    """qml.grad는 tuple을 돌려줄 수 있으므로 스칼라로 정규화."""
    g = qml.grad(qnode)(qml.numpy.array(x, requires_grad=True))
    return float(np.atleast_1d(np.asarray(g, dtype=float)).ravel()[0])

print("=" * 60)
print("환경")
print("=" * 60)
print(f"  torch     {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  gpu       {torch.cuda.get_device_name(0)}")
print(f"  pennylane {qml.__version__}")
print(f"  numpy     {np.__version__}")

dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev, interface="autograd")
def no_mixing(gamma):
    """RZZ 직후 바로 Z 측정 -> gradient 0이어야 함."""
    qml.RY(0.7, wires=0)
    qml.RY(0.4, wires=1)
    qml.IsingZZ(gamma, wires=[0, 1])
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev, interface="autograd")
def with_mixing(gamma):
    """RZZ -> local RX/RY mixing -> Z 측정 -> gradient 비영이어야 함."""
    qml.RY(0.7, wires=0)
    qml.RY(0.4, wires=1)
    qml.IsingZZ(gamma, wires=[0, 1])
    qml.RX(0.9, wires=0)   # 비가환 local mixing
    qml.RY(0.5, wires=0)
    return qml.expval(qml.PauliZ(0))


print()
print("=" * 60)
print("부록 B-1 / B-2: RZZ gradient가 local mixing에 의존하는가")
print("=" * 60)

g = 1.1
grad_no = scalar_grad(no_mixing, g)
grad_yes = scalar_grad(with_mixing, g)

print(f"  RZZ -> Z            : <Z0>={float(no_mixing(g)):+.6f}  d<Z0>/dgamma = {grad_no:+.3e}")
print(f"  RZZ -> RX/RY -> Z   : <Z0>={float(with_mixing(g)):+.6f}  d<Z0>/dgamma = {grad_yes:+.3e}")
print()

ok_b1 = abs(grad_no) < 1e-10
ok_b2 = abs(grad_yes) > 1e-6
print(f"  [B-1] mixing 없으면 gradient == 0 : {'PASS' if ok_b1 else 'FAIL'}")
print(f"  [B-2] mixing 있으면 gradient != 0 : {'PASS' if ok_b2 else 'FAIL'}")
print()
print("  -> v2 설계(RZZ 뒤 비가환 local mixing 필수)의 근거가 수치로 확인됨.")
print("     mixing을 빼면 마지막 얽힘층의 파라미터가 학습되지 않는다.")

print()
print("=" * 60)
print("B-3: torch statevector 구현이 PennyLane과 일치하는가")
print("=" * 60)


def torch_statevector(gamma: float) -> float:
    """with_mixing과 동일한 회로를 torch로 직접 구현."""
    c64 = torch.complex64

    def ry(t):
        c, s = np.cos(t / 2), np.sin(t / 2)
        return torch.tensor([[c, -s], [s, c]], dtype=c64)

    def rx(t):
        c, s = np.cos(t / 2), np.sin(t / 2)
        return torch.tensor([[c, -1j * s], [-1j * s, c]], dtype=c64)

    psi = torch.zeros(4, dtype=c64)
    psi[0] = 1.0

    def apply_1q(psi, U, q):
        # (2,2) on qubit q of a 2-qubit state; qubit 0 = most significant
        psi = psi.reshape(2, 2)
        if q == 0:
            psi = torch.einsum("ij,jk->ik", U, psi)
        else:
            psi = torch.einsum("ij,kj->ki", U, psi)
        return psi.reshape(4)

    psi = apply_1q(psi, ry(0.7), 0)
    psi = apply_1q(psi, ry(0.4), 1)

    # RZZ: diagonal phase exp(-i*gamma/2 * z_u z_v), z = +1/-1 by basis parity
    zz = torch.tensor([1.0, -1.0, -1.0, 1.0])  # (00,01,10,11) -> z0*z1
    psi = psi * torch.exp(-1j * torch.tensor(gamma / 2.0, dtype=c64) * zz)

    psi = apply_1q(psi, rx(0.9), 0)
    psi = apply_1q(psi, ry(0.5), 0)

    probs = (psi.conj() * psi).real
    z0 = torch.tensor([1.0, 1.0, -1.0, -1.0])  # <Z0>
    return float((probs * z0).sum())


pl_val = float(with_mixing(g))
tv_val = torch_statevector(g)
diff = abs(pl_val - tv_val)
print(f"  PennyLane <Z0> = {pl_val:+.9f}")
print(f"  torch     <Z0> = {tv_val:+.9f}")
print(f"  |diff|         = {diff:.2e}")
print(f"  [B-3] 두 구현 일치 : {'PASS' if diff < 1e-5 else 'FAIL'}")

print()
print("=" * 60)
all_ok = ok_b1 and ok_b2 and diff < 1e-5
print("전체:", "PASS — 환경과 회로 규약이 모두 검증됨" if all_ok else "FAIL")
print("=" * 60)
