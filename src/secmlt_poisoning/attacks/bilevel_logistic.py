"""Gradient-based (bilevel) poisoning attack against logistic regression.

Port of ``secml.adv.attacks.poisoning.CAttackPoisoningLogisticRegression``.

References
----------
- B. Biggio, B. Nelson, P. Laskov. Poisoning attacks against SVMs. ICML 2012.
- S. Mei, X. Zhu. Using machine teaching to identify optimal training-set
  attacks on machine learners. AAAI 2015.
- A. Demontis et al. Why do adversarial attacks transfer? Explaining
  transferability of evasion and poisoning attacks. USENIX Security 2019.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch

from secmlt_poisoning.attacks.base import (
    BilevelPoisoningAttack,
    PoisoningResult,
    poisoned_dataset,
)
from secmlt_poisoning.hypergradient import (
    implicit_hypergradient,
    logistic_hypergradient,
    mean_logistic_loss,
)
from secmlt_poisoning.models.logistic_regression import (
    LogisticRegression,
    fit_logistic_regression,
    logistic_objective,
    to_pm1,
)


class BilevelPoisoningLogisticRegression(BilevelPoisoningAttack):
    """Bilevel poisoning attack against binary L2 logistic regression.

    Solves::

        max_{x_p in [lb, ub]}  A(theta*(x_p))   (indiscriminate, ``y_target=None``)
        min_{x_p in [lb, ub]}  A_t(theta*(x_p)) (targeted, error-specific)

    where ``theta*`` is the LR trained on ``D_tr U {(x_p, y_p)}`` and A is the
    average logistic loss on the attacker's validation set. Labels ``y_p`` are
    fixed at initialization (label flipping), only features are optimized.

    All poisoning points are updated jointly: ``H^{-1} grad_theta A`` is shared
    among them, so the cost per iteration is one retraining + one (d+1)x(d+1)
    linear system, independently of the number of points. (secml optimized
    points one at a time.)

    Parameters
    ----------
    C : float
        Inverse regularization of the (surrogate) logistic regression.
    hypergradient : {"closed_form", "autograd"}
        Closed-form LR expression or generic autograd implicit differentiation.
    **kwargs
        See :class:`BilevelPoisoningAttack` (lb, ub, step_size, max_iter,
        y_target, line_search, tol, random_seed, store_path).
    """

    def __init__(
        self,
        C: float = 1.0,
        hypergradient: Literal["closed_form", "autograd"] = "closed_form",
        outer_loss: Callable = mean_logistic_loss,
        **kwargs,
    ) -> None:
        super().__init__(outer_loss=outer_loss, **kwargs)
        self.C = C
        self.hypergradient = hypergradient

    def _fit(self, x_tr, y_tr, x_p, y_p):
        x = torch.cat([self._flat(x_tr), self._flat(x_p)])
        y = torch.cat([to_pm1(y_tr), to_pm1(y_p)])
        w0, b0 = self._warm if self._warm is not None else (None, None)
        w, b = fit_logistic_regression(x, y, C=self.C, w0=w0, b0=b0)
        self._warm = (w, b)
        return w, b

    def _outer(self, state, x_val, y_val):
        w, b = state
        return self.outer_loss(x_val @ w + b, y_val)

    def _hypergradient(self, state, x_tr, y_tr, x_p, y_p, x_val, y_val):
        w, b = state
        if self.hypergradient == "closed_form":
            return logistic_hypergradient(
                w, b, x_tr, y_tr, x_p, y_p, x_val, y_val, self.C, outer_loss=self.outer_loss
            )
        if self.hypergradient == "autograd":
            return self._autograd_hypergradient(w, b, x_tr, y_tr, x_p, y_p, x_val, y_val)
        msg = f"Unknown hypergradient method {self.hypergradient!r}."
        raise ValueError(msg)

    def _autograd_hypergradient(self, w, b, x_tr, y_tr, x_p, y_p, x_val, y_val):
        x_tr = self._flat(x_tr)
        y_tr, y_p = to_pm1(y_tr), to_pm1(y_p)
        d = x_tr.shape[1]

        def inner(theta, xp):
            x = torch.cat([x_tr, xp.reshape(xp.shape[0], -1)])
            return logistic_objective(theta[:d], theta[d], x, torch.cat([y_tr, y_p]), self.C)

        def outer(theta):
            return self.outer_loss(x_val @ theta[:d] + theta[d], y_val)

        theta = torch.cat([w, b.reshape(1)])
        return implicit_hypergradient(inner, outer, theta, x_p.to(torch.float64))

    def _to_model(self, state):
        w, b = state
        return LogisticRegression(w.numel(), C=self.C).set_params(w, b)


__all__ = ["BilevelPoisoningLogisticRegression", "PoisoningResult", "poisoned_dataset"]
