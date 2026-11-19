"""Binary L2-regularized logistic regression trained to (numerical) optimality.

The training problem is the one used in secml / LIBLINEAR / scikit-learn::

    min_{w, b}  1/2 ||w||^2 + C * sum_i log(1 + exp(-y_i (w^T x_i + b)))

with labels ``y_i`` in {-1, +1} (the bias is *not* regularized).

Bilevel poisoning relies on the implicit function theorem, which assumes that
the inner problem is solved exactly (gradient = 0). For this reason we solve it
with a damped Newton method in float64 rather than with SGD.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def to_pm1(y: torch.Tensor) -> torch.Tensor:
    """Map labels in {0, 1} to {-1, +1} (labels already in {-1, +1} are kept)."""
    y = torch.as_tensor(y)
    if y.min() < 0:
        return y.to(torch.float64)
    return 2.0 * y.to(torch.float64) - 1.0


def logistic_objective(
    w: torch.Tensor, b: torch.Tensor, x: torch.Tensor, y: torch.Tensor, C: float
) -> torch.Tensor:
    """Regularized training objective (labels in {-1, +1})."""
    s = x @ w + b
    return 0.5 * w @ w + C * nn.functional.softplus(-y * s).sum()


def fit_logistic_regression(
    x: torch.Tensor,
    y: torch.Tensor,
    C: float = 1.0,
    w0: torch.Tensor | None = None,
    b0: torch.Tensor | float | None = None,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train binary logistic regression with damped Newton's method.

    Parameters
    ----------
    x : torch.Tensor
        Training samples, shape (n, d) (flattened internally).
    y : torch.Tensor
        Labels in {0, 1} or {-1, +1}.
    C : float
        Inverse regularization strength.
    w0, b0 : optional
        Warm start (useful inside the attack loop, where the training set
        changes only slightly between two consecutive iterations).
    max_iter : int
        Maximum number of Newton iterations.
    tol : float
        Stop when the infinity-norm of the gradient is below ``tol``.

    Returns
    -------
    (w, b) : tuple of torch.Tensor
        Optimal weights (d,) and bias (scalar tensor), float64.
    """
    x = x.reshape(x.shape[0], -1).to(torch.float64)
    y = to_pm1(y).to(x.device)
    n, d = x.shape
    xa = torch.cat([x, torch.ones(n, 1, dtype=x.dtype, device=x.device)], dim=1)
    theta = torch.zeros(d + 1, dtype=x.dtype, device=x.device)
    if w0 is not None:
        theta[:d] = w0.detach().to(theta)
    if b0 is not None:
        theta[d] = torch.as_tensor(b0).detach().to(theta)
    reg = torch.ones(d + 1, dtype=x.dtype, device=x.device)
    reg[d] = 0.0  # bias is not regularized

    def obj(t: torch.Tensor) -> torch.Tensor:
        return logistic_objective(t[:d], t[d], x, y, C)

    f = obj(theta)
    for _ in range(max_iter):
        s = xa @ theta
        p = torch.sigmoid(y * s)  # prob. of the correct class
        grad = reg * theta + C * xa.T @ (-y * (1 - p))
        if grad.abs().max() < tol:
            break
        z = p * (1 - p)
        hess = C * xa.T @ (z[:, None] * xa) + torch.diag(reg)
        # tiny jitter keeps the bias block PD if all z ~ 0 (separable data)
        hess.diagonal().add_(1e-12)
        step = torch.linalg.solve(hess, grad)
        # backtracking (Armijo) line search
        t, dec = 1.0, (grad @ step).item()
        # close to the optimum (tiny Newton decrement) objective differences
        # are below float precision: take the full (quadratically convergent)
        # step instead of letting Armijo fail on round-off.
        theta_new = theta - step
        f_new = obj(theta_new)
        while dec > 1e-14 and f_new > f - 1e-4 * t * dec and t > 1e-10:
            t *= 0.5
            theta_new = theta - t * step
            f_new = obj(theta_new)
        theta, f = theta_new, f_new
    return theta[:d].clone(), theta[d].clone()


class LogisticRegression(nn.Module):
    """Binary logistic regression as a ``torch.nn.Module``.

    ``forward`` returns two logits ``[0, f(x)]`` with ``f(x) = w^T x + b``, so
    that ``argmax`` gives the predicted class in {0, 1} and ``softmax`` gives
    ``[1 - sigmoid(f), sigmoid(f)]``. This makes the model directly usable with
    :class:`secmlt.models.pytorch.BasePyTorchClassifier` and secml-torch
    metrics.
    """

    def __init__(self, n_features: int, C: float = 1.0) -> None:
        super().__init__()
        self.C = C
        self.weight = nn.Parameter(torch.zeros(n_features, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float64))

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminant function f(x) = w^T x + b, shape (n,)."""
        x = x.reshape(x.shape[0], -1).to(self.weight.dtype)
        return x @ self.weight + self.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        f = self.score(x)
        return torch.stack([torch.zeros_like(f), f], dim=-1)

    @torch.no_grad()
    def set_params(self, w: torch.Tensor, b: torch.Tensor) -> LogisticRegression:
        """Copy parameters (e.g. the output of ``fit_logistic_regression``)."""
        self.weight.copy_(w)
        self.bias.copy_(b)
        return self

    def fit(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> LogisticRegression:
        """Train in place on tensors (labels in {0,1} or {-1,+1})."""
        w, b = fit_logistic_regression(x, y, C=self.C, **kwargs)
        return self.set_params(w, b)


class LogisticRegressionTrainer:
    """secml-torch compatible trainer: ``trainer.train(model, dataloader)``.

    Collects the whole dataloader into a single batch and solves the training
    problem exactly with Newton's method. Plug it into
    ``BasePyTorchClassifier(model, trainer=LogisticRegressionTrainer())``.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-10) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def train(self, model: LogisticRegression, dataloader: DataLoader) -> nn.Module:
        """Train ``model`` on all data in ``dataloader``."""
        xs, ys = zip(*dataloader)
        x, y = torch.cat(xs), torch.cat(ys)
        return model.fit(x, y, max_iter=self.max_iter, tol=self.tol)
