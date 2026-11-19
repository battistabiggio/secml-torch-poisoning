"""Shared machinery for bilevel poisoning attacks.

Subclasses implement the learner-specific parts:

* ``_fit(x_tr, y_tr, x_p, y_p)``  -> state of the trained (poisoned) learner
* ``_outer(state, x_val, y_val_pm1)`` -> attacker loss A
* ``_hypergradient(state, x_tr, y_tr, x_p, y_p, x_val, y_val_pm1)`` -> (dA/dx_p, A)
* ``_to_model(state)`` -> ``nn.Module`` usable with secml-torch

The base class provides initialization (random points with flipped labels),
box projection and projected gradient ascent/descent with line search.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from secmlt_poisoning.models.logistic_regression import to_pm1

logger = logging.getLogger(__name__)


@dataclass
class PoisoningResult:
    """Output of a poisoning attack."""

    x_poison: torch.Tensor
    y_poison: torch.Tensor
    model: nn.Module
    history: dict = field(default_factory=dict)


class BilevelPoisoningAttack(ABC):
    """Gradient-based poisoning: projected gradient on the poisoning features.

    Parameters
    ----------
    lb, ub : float or torch.Tensor
        Box constraint on the poisoning features.
    step_size : float
        Initial (and maximum) step. Gradients are normalized per point, so each
        point moves by at most ``step_size`` (L2) per iteration.
    max_iter : int
        Maximum number of iterations (0 = return the classifier trained on the
        random label-flip initialization).
    y_target : int, torch.Tensor or None
        None: indiscriminate attack (maximize A on the validation labels).
        Otherwise error-specific attack (minimize A w.r.t. these labels).
    line_search : bool
        Halve the step until the objective improves (as secml's ``pgd-ls``).
    tol : float
        Stop when the objective improvement is below ``tol``.
    outer_loss : callable
        ``outer_loss(scores, y_pm1) -> scalar``.
    random_seed : int or None
        Seed for the initialization.
    store_path : bool
        Store the poisoning points at every iteration in ``history["x_poison"]``.
    """

    def __init__(
        self,
        outer_loss: Callable,
        lb: float | torch.Tensor = 0.0,
        ub: float | torch.Tensor = 1.0,
        step_size: float = 0.1,
        max_iter: int = 100,
        y_target: int | torch.Tensor | None = None,
        line_search: bool = True,
        tol: float = 1e-6,
        random_seed: int | None = None,
        store_path: bool = False,
    ) -> None:
        self.outer_loss = outer_loss
        self.lb, self.ub = lb, ub
        self.step_size = step_size
        self.max_iter = max_iter
        self.y_target = y_target
        self.line_search = line_search
        self.tol = tol
        self.random_seed = random_seed
        self.store_path = store_path
        self._warm: Any = None

    # --------------------------------------------------------- to implement
    @abstractmethod
    def _fit(self, x_tr, y_tr, x_p, y_p) -> Any: ...

    @abstractmethod
    def _outer(self, state, x_val, y_val) -> torch.Tensor: ...

    @abstractmethod
    def _hypergradient(self, state, x_tr, y_tr, x_p, y_p, x_val, y_val): ...

    @abstractmethod
    def _to_model(self, state) -> nn.Module: ...

    # ---------------------------------------------------------------- utils
    @staticmethod
    def _flat(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1).to(torch.float64)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        lb = torch.as_tensor(self.lb, dtype=x.dtype)
        ub = torch.as_tensor(self.ub, dtype=x.dtype)
        return torch.maximum(torch.minimum(x, ub), lb)

    def _val_labels(self, y_val: torch.Tensor) -> torch.Tensor:
        if self.y_target is None:
            return to_pm1(y_val)
        yt = torch.as_tensor(self.y_target)
        return to_pm1(yt.expand(y_val.shape[0]) if yt.ndim == 0 else yt)

    @property
    def _sign(self) -> float:
        """+1 to maximize (indiscriminate), -1 to minimize (targeted)."""
        return 1.0 if self.y_target is None else -1.0

    def init_poisoning_points(
        self, x: torch.Tensor, y: torch.Tensor, n_points: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``n_points`` from (x, y), flip their labels, add tiny noise."""
        gen = (
            torch.Generator().manual_seed(self.random_seed)
            if self.random_seed is not None
            else None
        )
        idx = torch.randperm(x.shape[0], generator=gen)[:n_points]
        x_p = x[idx].to(torch.float64).clone()
        x_p += 1e-3 * (2 * torch.rand(x_p.shape, generator=gen, dtype=x_p.dtype) - 1)
        x_p = self._project(x_p)
        y = torch.as_tensor(y)[idx]
        y_p = -y if y.min() < 0 else 1 - y
        return x_p, y_p

    # ------------------------------------------------------- public helpers
    def train(self, x_tr, y_tr, x_p, y_p, warm_start: bool = True):
        """Train the learner on ``D_tr U D_p`` (warm-started by default)."""
        if not warm_start:
            self._warm = None
        state = self._fit(x_tr, y_tr, x_p, y_p)
        return state

    def objective(self, x_tr, y_tr, x_p, y_p, x_val, y_val) -> torch.Tensor:
        """Attacker loss A(theta*(x_p))."""
        state = self.train(x_tr, y_tr, x_p, y_p)
        return self._outer(state, self._flat(x_val), self._val_labels(y_val)).detach()

    def gradient(self, x_tr, y_tr, x_p, y_p, x_val, y_val):
        """Hypergradient dA/dx_p (after retraining). Returns (grad, A)."""
        state = self.train(x_tr, y_tr, x_p, y_p)
        g, a = self._hypergradient(
            state, x_tr, y_tr, x_p, y_p, self._flat(x_val), self._val_labels(y_val)
        )
        return g.reshape(x_p.shape), a

    # ------------------------------------------------------------------ run
    def run(
        self,
        x_tr: torch.Tensor,
        y_tr: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        n_points: int | None = None,
        x_init: torch.Tensor | None = None,
        y_init: torch.Tensor | None = None,
        callback: Callable | None = None,
    ) -> PoisoningResult:
        """Run the attack.

        Either pass ``n_points`` (points are sampled from the training set and
        their labels are flipped) or an explicit initialization
        ``(x_init, y_init)``. ``callback(it, x_p, objective)`` is called after
        every iteration.
        """
        if x_init is None:
            if not n_points:
                msg = "Pass n_points > 0 or (x_init, y_init)."
                raise ValueError(msg)
            x_p, y_p = self.init_poisoning_points(x_tr, y_tr, n_points)
        else:
            x_p = self._project(x_init.to(torch.float64).clone())
            y_p = torch.as_tensor(y_init).clone()

        self._warm = None
        sign = self._sign
        eta = self.step_size
        grad, obj = self.gradient(x_tr, y_tr, x_p, y_p, x_val, y_val)
        history = {"objective": [obj.item()], "step_size": [], "x_poison": []}
        if self.store_path:
            history["x_poison"].append(x_p.clone())

        for it in range(self.max_iter):
            g = sign * grad.reshape(grad.shape[0], -1)
            if g.abs().max() == 0:
                logger.info("iter %d: zero gradient, stopping.", it)
                break
            g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
            direction = g.reshape(x_p.shape)

            accepted = False
            while True:
                x_new = self._project(x_p + eta * direction)
                obj_new = self.objective(x_tr, y_tr, x_new, y_p, x_val, y_val)
                if not self.line_search or sign * (obj_new - obj) > 0:
                    accepted = True
                    break
                eta *= 0.5
                if eta < 1e-6 * self.step_size:
                    break
            if not accepted:
                logger.info("iter %d: no improving step found, stopping.", it)
                break

            improvement = sign * (obj_new - obj).item()
            x_p = x_new
            history["step_size"].append(eta)
            if self.line_search:  # let the step grow back after a success
                eta = min(2 * eta, self.step_size)
            grad, obj = self.gradient(x_tr, y_tr, x_p, y_p, x_val, y_val)
            history["objective"].append(obj.item())
            if self.store_path:
                history["x_poison"].append(x_p.clone())
            if callback is not None:
                callback(it, x_p, obj)
            logger.info("iter %d: objective %.6f", it, obj.item())
            if self.line_search and improvement < self.tol:
                break

        state = self.train(x_tr, y_tr, x_p, y_p, warm_start=False)
        return PoisoningResult(
            x_poison=x_p.detach(), y_poison=y_p, model=self._to_model(state), history=history
        )


def poisoned_dataset(
    x_tr: torch.Tensor, y_tr: torch.Tensor, result: PoisoningResult
) -> torch.utils.data.TensorDataset:
    """Concatenate clean training data and poisoning points as a dataset."""
    x_p = result.x_poison.reshape(-1, *x_tr.shape[1:]).to(x_tr.dtype)
    return torch.utils.data.TensorDataset(
        torch.cat([x_tr, x_p]), torch.cat([y_tr, result.y_poison.to(y_tr.dtype)])
    )
