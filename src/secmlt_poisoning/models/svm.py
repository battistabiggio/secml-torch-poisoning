r"""Binary (kernel) SVM in PyTorch with an exact dual solver.

Dual problem (labels in {-1, +1}, ``Q_ij = y_i y_j K(x_i, x_j)``)::

    min_alpha  1/2 alpha^T Q alpha - 1^T alpha
    s.t.       0 <= alpha_i <= C,   y^T alpha = 0

and ``f(x) = sum_j alpha_j y_j K(x, x_j) + b``.

Bilevel poisoning differentiates the KKT conditions restricted to the *margin*
support vectors, so it needs the exact partition of the training set into

* ``S``: margin SVs,   0 < alpha < C,  y f(x) = 1
* ``E``: error vectors, alpha = C,     y f(x) <= 1
* ``R``: reserve points, alpha = 0,    y f(x) >= 1

The solver is a primal active-set QP method (Nocedal & Wright, Alg. 16.3):
the free variables are the margin SVs, and each iteration solves the KKT
linear system in ``(alpha_S, b)`` exactly, so the partition it returns is
exact. A few FISTA iterations provide the cold start; inside the attack the
previous ``alpha`` is used as warm start, and only a few active-set changes
are needed per iteration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from secmlt_poisoning.models.logistic_regression import to_pm1


# --------------------------------------------------------------------- kernels
class LinearKernel:
    """K(a, b) = a^T b."""

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return a @ b.T

    def __repr__(self) -> str:  # noqa: D105
        return "LinearKernel()"


class RBFKernel:
    """K(a, b) = exp(-gamma ||a - b||^2)."""

    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: D102
        # explicit expansion keeps it differentiable also for a == b
        d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
        return torch.exp(-self.gamma * d2.clamp_min(0))

    def __repr__(self) -> str:  # noqa: D105
        return f"RBFKernel(gamma={self.gamma})"


# ---------------------------------------------------------------------- solver
@dataclass
class SVMDualSolution:
    """Solution of the SVM dual and partition of the training points."""

    alpha: torch.Tensor  # (n,)
    b: torch.Tensor  # scalar
    margin: torch.Tensor  # bool (n,), S
    error: torch.Tensor  # bool (n,), E
    kkt_violation: float
    converged: bool


def _project_box_hyperplane(z, y, C, iters: int = 100):
    """Euclidean projection onto {0 <= a <= C, y^T a = 0} (bisection on mu)."""
    hi = z.abs().max() + C
    lo = -hi
    for _ in range(iters):
        mu = 0.5 * (lo + hi)
        s = (y * (z - mu * y).clamp(0, C)).sum()
        # s is non-increasing in mu
        lo, hi = (mu, hi) if s > 0 else (lo, mu)
    return (z - 0.5 * (lo + hi) * y).clamp(0, C)


def _fista(Q, y, C, alpha0, n_iter):
    """Accelerated projected gradient on the dual (approximate, feasible)."""
    lip = torch.linalg.eigvalsh(Q)[-1].clamp_min(1e-12)
    a = alpha0.clone()
    z, t = a.clone(), 1.0
    for _ in range(n_iter):
        a_new = _project_box_hyperplane(z - (Q @ z - 1) / lip, y, C)
        t_new = 0.5 * (1 + (1 + 4 * t * t) ** 0.5)
        z = a_new + ((t - 1) / t_new) * (a_new - a)
        a, t = a_new, t_new
    return a


def kkt_matrix(Q: torch.Tensor, y: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Bordered KKT matrix ``[[Q_FF, y_F], [y_F^T, 0]]`` of the free variables."""
    f = F.nonzero().squeeze(1)
    m = f.numel()
    A = torch.zeros(m + 1, m + 1, dtype=Q.dtype)
    A[:m, :m] = Q[f][:, f]
    A[:m, m] = y[f]
    A[m, :m] = y[f]
    return A


def _target(Q, y, C, alpha, F, E):
    """Minimizer of the dual on the face {alpha_i fixed for i not in F}.

    Solves ``Q_FF a_F + y_F b = 1 - Q_FE C``, ``y_F^T a_F = -C sum_E y`` i.e.
    the margin conditions ``y_i f(x_i) = 1`` for the free points.
    """
    f = F.nonzero().squeeze(1)
    m = f.numel()
    A = kkt_matrix(Q, y, F)
    rhs = torch.cat([1 - Q[f][:, E] @ alpha[E], -(y[E] * C).sum().reshape(1)])
    sol = torch.linalg.solve(A, rhs)
    a = alpha.clone()
    a[f] = sol[:m]
    return a, sol[m]


def _best_bias(Q, y, alpha, at_zero, at_c):
    """Bias when no variable is free (b is then only bounded to an interval).

    KKT requires ``g_i(b) = (Q alpha)_i + y_i b - 1 >= 0`` at alpha=0 and
    ``<= 0`` at alpha=C, i.e. lower/upper bounds on b. We return the midpoint
    of the interval (as libsvm), or the best compromise if it is empty.
    """
    qa = Q @ alpha
    r = y * (1 - qa)  # g_i(b) = 0 at b = r_i
    pos, neg = y > 0, y < 0
    lower = torch.cat([r[at_zero & pos], r[at_c & neg]])  # b >= lower
    upper = torch.cat([r[at_zero & neg], r[at_c & pos]])  # b <= upper
    lo = lower.max() if lower.numel() else None
    hi = upper.min() if upper.numel() else None
    if lo is not None and hi is not None:
        return 0.5 * (lo + hi)
    return lo if lo is not None else (hi if hi is not None else torch.zeros((), dtype=Q.dtype))


def solve_svm_dual(
    K: torch.Tensor,
    y: torch.Tensor,
    C: float,
    alpha0: torch.Tensor | None = None,
    jitter: float = 1e-10,
    fista_iter: int = 100,
    max_iter: int = 5000,
    tol: float = 1e-9,
) -> SVMDualSolution:
    """Solve the SVM dual exactly with a primal active-set QP method.

    Nocedal & Wright, *Numerical Optimization*, Alg. 16.3, specialized to box
    constraints plus the equality ``y^T alpha = 0``. Each iteration computes
    the minimizer on the current face (one KKT linear system on the free
    variables); if it is feasible it is accepted and the multipliers of the
    bounded variables are checked, otherwise we step to the first blocking
    bound.

    Parameters
    ----------
    K : torch.Tensor
        Kernel matrix (n, n).
    y : torch.Tensor
        Labels in {-1, +1}.
    C : float
        Box constraint.
    alpha0 : torch.Tensor, optional
        Warm start (projected onto the feasible set). If None, a few FISTA
        iterations provide the starting point.
    jitter : float
        Relative ridge ``jitter * max(diag K)`` added to K. It makes the dual
        strictly convex (unique alpha, bounded subproblems) also for singular
        kernel matrices (linear kernel with n > d, RBF with large gamma^-1,
        duplicated points).
    """
    K = K.to(torch.float64)
    y = y.to(torch.float64)
    n = y.shape[0]
    Q = (y[:, None] * y[None, :]) * K
    Q = Q + jitter * K.diagonal().abs().max().clamp_min(1e-12) * torch.eye(n, dtype=Q.dtype)

    if alpha0 is None:
        alpha = _fista(Q, y, C, torch.full((n,), C / 2, dtype=Q.dtype), fista_iter)
    else:
        alpha = _project_box_hyperplane(alpha0.to(torch.float64), y, C)
    # working set: variables fixed at a bound
    at_zero = alpha <= 1e-12 * C
    at_c = alpha >= C * (1 - 1e-12)
    alpha = torch.where(at_zero, 0.0, torch.where(at_c, float(C), alpha))

    b = torch.zeros((), dtype=Q.dtype)
    converged, viol = False, float("inf")
    for _ in range(max_iter):
        F = ~at_zero & ~at_c
        if F.any():
            target, b_t = _target(Q, y, C, alpha, F, at_c)
            p = target - alpha
        else:
            target, b_t, p = alpha, None, torch.zeros_like(alpha)
        feasible = bool(((target[F] >= 0) & (target[F] <= C)).all())
        if feasible:
            # minimizer on the face: accept it and check the bound multipliers
            alpha = target
            b = b_t if b_t is not None else _best_bias(Q, y, alpha, at_zero, at_c)
            g = Q @ alpha - 1 + y * b  # = y_i f(x_i) - 1
            v = torch.zeros(n, dtype=Q.dtype)
            v[at_zero] = (-g[at_zero]).clamp_min(0)  # need g >= 0 at alpha=0
            v[at_c] = g[at_c].clamp_min(0)  # need g <= 0 at alpha=C
            viol = v.max().item()
            if viol <= tol:
                converged = True
                break
            i = v.argmax()
            at_zero[i] = at_c[i] = False  # release the most violated bound
            continue
        # step towards the face minimizer, stop at the first blocking bound
        t = torch.full_like(alpha, float("inf"))
        neg, pos = F & (target < 0), F & (target > C)
        t[neg] = -alpha[neg] / p[neg]
        t[pos] = (C - alpha[pos]) / p[pos]
        t_block, i = t.min(0)
        alpha = alpha + t_block.clamp(0, 1) * p
        if p[i] < 0:
            alpha[i], at_zero[i] = 0.0, True
        else:
            alpha[i], at_c[i] = float(C), True

    alpha = alpha.clamp(0, C)
    S = ~at_zero & ~at_c
    if not converged and not S.any():
        b = _best_bias(Q, y, alpha, at_zero, at_c)
    return SVMDualSolution(alpha, b, S, at_c.clone(), viol, converged)


# ----------------------------------------------------------------------- model
class SVM(nn.Module):
    """Binary kernel SVM as ``torch.nn.Module``.

    ``forward`` returns ``[0, f(x)]`` (argmax -> class in {0, 1}), compatible
    with :class:`secmlt.models.pytorch.BasePyTorchClassifier`.
    """

    def __init__(self, C: float = 1.0, kernel=None, jitter: float = 1e-10) -> None:
        super().__init__()
        self.C = C
        self.jitter = jitter
        self.kernel = kernel if kernel is not None else LinearKernel()
        self.register_buffer("x_train", torch.zeros(0, 0, dtype=torch.float64))
        self.register_buffer("coef", torch.zeros(0, dtype=torch.float64))  # alpha * y
        # a (frozen) parameter, so that secml-torch can infer the device
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float64), requires_grad=False)
        self.solution: SVMDualSolution | None = None

    def fit(
        self, x: torch.Tensor, y: torch.Tensor, alpha0: torch.Tensor | None = None
    ) -> SVM:
        """Train on tensors (labels in {0,1} or {-1,+1})."""
        x = x.reshape(x.shape[0], -1).to(torch.float64)
        y = to_pm1(y)
        sol = solve_svm_dual(self.kernel(x, x), y, self.C, alpha0=alpha0, jitter=self.jitter)
        return self.set_solution(x, y, sol)

    def set_solution(self, x, y, sol: SVMDualSolution) -> SVM:
        """Set training data and dual solution."""
        self.x_train, self.coef = x, sol.alpha * y
        self.bias.data = sol.b.detach().clone().reshape(())
        self.solution = sol
        return self

    @property
    def weight(self) -> torch.Tensor:
        """Primal weights (linear kernel only)."""
        return self.coef @ self.x_train

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminant function f(x), shape (n,)."""
        x = x.reshape(x.shape[0], -1).to(torch.float64)
        return self.kernel(x, self.x_train) @ self.coef + self.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        f = self.score(x)
        return torch.stack([torch.zeros_like(f), f], dim=-1)


class SVMTrainer:
    """secml-torch compatible trainer (full batch, exact dual solver)."""

    def train(self, model: SVM, dataloader) -> nn.Module:  # noqa: D102
        xs, ys = zip(*dataloader)
        return model.fit(torch.cat(xs), torch.cat(ys))
