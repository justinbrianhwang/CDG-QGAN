"""Environment smoke test + verification of the core circuit claims in Appendix B of the plan.

Checks:
  B-1. Z measurement immediately after RZZ -> the entangling gradient is 0 (RZZ commutes with Z)
  B-2. RX/RY mixing after RZZ -> the gradient is nonzero
  B-3. The PennyLane and torch statevector implementations agree
"""

import sys

import numpy as np
import pennylane as qml
import torch

sys.stdout.reconfigure(encoding="utf-8")


def scalar_grad(qnode, x):
    """qml.grad may return a tuple, so normalize it to a scalar."""
    g = qml.grad(qnode)(qml.numpy.array(x, requires_grad=True))
    return float(np.atleast_1d(np.asarray(g, dtype=float)).ravel()[0])

print("=" * 60)
print("Environment")
print("=" * 60)
print(f"  torch     {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  gpu       {torch.cuda.get_device_name(0)}")
print(f"  pennylane {qml.__version__}")
print(f"  numpy     {np.__version__}")

dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev, interface="autograd")
def no_mixing(gamma):
    """Z measured immediately after RZZ -> the gradient must be 0."""
    qml.RY(0.7, wires=0)
    qml.RY(0.4, wires=1)
    qml.IsingZZ(gamma, wires=[0, 1])
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev, interface="autograd")
def with_mixing(gamma):
    """RZZ -> local RX/RY mixing -> Z measurement -> the gradient must be nonzero."""
    qml.RY(0.7, wires=0)
    qml.RY(0.4, wires=1)
    qml.IsingZZ(gamma, wires=[0, 1])
    qml.RX(0.9, wires=0)   # non-commuting local mixing
    qml.RY(0.5, wires=0)
    return qml.expval(qml.PauliZ(0))


print()
print("=" * 60)
print("Appendix B-1 / B-2: does the RZZ gradient depend on local mixing?")
print("=" * 60)

g = 1.1
grad_no = scalar_grad(no_mixing, g)
grad_yes = scalar_grad(with_mixing, g)

print(f"  RZZ -> Z            : <Z0>={float(no_mixing(g)):+.6f}  d<Z0>/dgamma = {grad_no:+.3e}")
print(f"  RZZ -> RX/RY -> Z   : <Z0>={float(with_mixing(g)):+.6f}  d<Z0>/dgamma = {grad_yes:+.3e}")
print()

ok_b1 = abs(grad_no) < 1e-10
ok_b2 = abs(grad_yes) > 1e-6
print(f"  [B-1] without mixing, gradient == 0 : {'PASS' if ok_b1 else 'FAIL'}")
print(f"  [B-2] with mixing, gradient != 0    : {'PASS' if ok_b2 else 'FAIL'}")
print()
print("  -> The rationale for the v2 design (non-commuting local mixing after RZZ is mandatory) is confirmed numerically.")
print("     Without the mixing, the parameters of the final entangling layer do not train.")

print()
print("=" * 60)
print("B-3: does the torch statevector implementation agree with PennyLane?")
print("=" * 60)


def torch_statevector(gamma: float) -> float:
    """The same circuit as with_mixing, implemented directly in torch."""
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
print(f"  [B-3] the two implementations agree : {'PASS' if diff < 1e-5 else 'FAIL'}")

print()
print("=" * 60)
all_ok = ok_b1 and ok_b2 and diff < 1e-5
print("Overall:", "PASS — environment and circuit conventions all verified" if all_ok else "FAIL")
print("=" * 60)
