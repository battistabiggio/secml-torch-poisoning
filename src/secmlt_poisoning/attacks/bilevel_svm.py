"""Gradient-based (bilevel) poisoning attack against (kernel) SVMs.

Port of ``secml.adv.attacks.poisoning.CAttackPoisoningSVM``.

References
----------
- B. Biggio, B. Nelson, P. Laskov. Poisoning attacks against support vector
  machines. ICML 2012.
- G. Cauwenberghs, T. Poggio. Incremental and decremental support vector
  machine learning. NIPS 2000 (adiabatic update of the SVM solution).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from secmlt_poisoning.attacks.base import BilevelPoisoningAttack
from secmlt_poisoning.hypergradient import mean_hinge_loss, svm_hypergradient
from secmlt_poisoning.models.logistic_regression import to_pm1
from secmlt_poisoning.models.svm import SVM, LinearKernel


class BilevelPoisoningSVM(BilevelPoisoningAttack):
    """Bilevel poisoning attack against a binary (kernel) SVM.

    The inner problem is solved exactly with an active-set dual solver; the
    hypergradient is obtained by implicit differentiation of the KKT
    conditions on the margin support vectors (no second derivative of the
    hinge loss is needed). See :func:`secmlt_poisoning.svm_hypergradient`.

    Notes
    -----
    * The gradient is zero for poisoning points with alpha_c = 0 (they do not
      influence the SVM). The label-flip initialization usually makes them
      error vectors (alpha_c = C).
    * The attacker objective is continuous but only piecewise smooth: its
      gradient jumps when a point changes group (S, E, R). The line search
      handles these kinks.

    Parameters
    ----------
    C : float
        SVM regularization parameter.
    kernel : callable
        ``LinearKernel()`` (default) or ``RBFKernel(gamma)`` or any
        differentiable ``kernel(A, B)``.
    outer_loss : callable
        Attacker loss on validation scores. Default: mean hinge loss, as secml.
    **kwargs
        See :class:`BilevelPoisoningAttack`.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel=None,
        outer_loss: Callable = mean_hinge_loss,
        jitter: float = 1e-10,
        **kwargs,
    ) -> None:
        super().__init__(outer_loss=outer_loss, **kwargs)
        self.C = C
        self.kernel = kernel if kernel is not None else LinearKernel()
        self.jitter = jitter

    def _fit(self, x_tr, y_tr, x_p, y_p):
        x = torch.cat([self._flat(x_tr), self._flat(x_p)])
        y = torch.cat([to_pm1(y_tr), to_pm1(y_p)])
        alpha0 = self._warm if self._warm is not None and self._warm.numel() == len(y) else None
        model = SVM(self.C, self.kernel, jitter=self.jitter).fit(x, y, alpha0=alpha0)
        if not model.solution.converged:
            model = SVM(self.C, self.kernel, jitter=self.jitter).fit(x, y)  # cold restart
        self._warm = model.solution.alpha
        return model

    def _outer(self, state, x_val, y_val):
        return self.outer_loss(state.score(x_val), y_val)

    def _hypergradient(self, state, x_tr, y_tr, x_p, y_p, x_val, y_val):
        return svm_hypergradient(
            state.solution, self.kernel, x_tr, y_tr, x_p, y_p, x_val, y_val, self.C,
            outer_loss=self.outer_loss, jitter=self.jitter,
        )

    def _to_model(self, state):
        return state


__all__ = ["BilevelPoisoningSVM"]
