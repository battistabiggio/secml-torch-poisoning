r"""Hypergradients for bilevel poisoning via implicit differentiation.

Bilevel problem (indiscriminate poisoning)::

    max_{x_c}  A(theta*(x_c))                     outer: attacker loss on D_val
    s.t.       theta*(x_c) = argmin_theta L(theta; D_tr U {(x_c, y_c)})

Since ``grad_theta L(theta*, x_c) = 0`` at the inner optimum, the implicit
function theorem gives ``d theta*/d x_c = - H^{-1} d(grad_theta L)/d x_c`` and

.. math::

    \nabla_{x_c} A = - \frac{\partial^2 L}{\partial x_c \partial \theta^\top}
                       H^{-1} \nabla_\theta A,
    \qquad H = \nabla^2_\theta L .

Computing ``v = H^{-1} grad_theta A`` once (one linear system, independent of
the number of poisoning points) the gradient for *every* poisoning point is
``-d/dx_c [ v^T grad_theta L(theta, x_c) ]``, i.e. a single
Jacobian-vector product.

Logistic regression, closed form
--------------------------------
With ``L = 1/2 ||w||^2 + C sum_i log(1 + exp(-y_i (w^T x_i + b)))``,
``sigma_i = sigmoid(y_i s_i)``, ``s_i = w^T x_i + b``, ``z_i = sigma_i (1 - sigma_i)``:

* Hessian (bias unregularized)::

      H = [[I, 0], [0, 0]] + C sum_i z_i [x_i; 1] [x_i; 1]^T

* Mixed second derivative for poisoning point ``c`` (d x (d+1))::

      G_c = d(grad_theta L)/dx_c
          = C [ -y_c (1 - sigma_c) [I, 0] + z_c w [x_c; 1]^T ]

* Hypergradient::

      grad_{x_c} A = - G_c v = C [ y_c (1 - sigma_c) v_w - z_c (x_c^T v_w + v_b) w ]

  with ``v = [v_w; v_b] = H^{-1} grad_theta A``.

This matches ``CAttackPoisoningLogisticRegression._gradient_fk_xc`` in secml
(Demontis et al., USENIX Security 2019, Sect. 3.3), but avoids forming G and
H^{-1} explicitly.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from secmlt_poisoning.models.logistic_regression import to_pm1


def mean_logistic_loss(scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Average logistic loss ``log(1 + exp(-y f))`` (labels in {-1, +1})."""
    return nn.functional.softplus(-y * scores).mean()


def outer_gradient(
    w: torch.Tensor,
    b: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    outer_loss: Callable = mean_logistic_loss,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the attacker loss A(theta) and its gradient w.r.t. theta=[w; b]."""
    theta = torch.cat([w.detach(), b.detach().reshape(1)]).requires_grad_()
    d = w.numel()
    loss = outer_loss(x_val @ theta[:d] + theta[d], y_val)
    (g,) = torch.autograd.grad(loss, theta)
    return loss.detach(), g


def logistic_hessian(
    w: torch.Tensor, b: torch.Tensor, x: torch.Tensor, y: torch.Tensor, C: float
) -> torch.Tensor:
    """Hessian of the LR training objective w.r.t. theta = [w; b]."""
    n, d = x.shape
    xa = torch.cat([x, torch.ones(n, 1, dtype=x.dtype, device=x.device)], dim=1)
    p = torch.sigmoid(y * (x @ w + b))
    z = p * (1 - p)
    hess = C * xa.T @ (z[:, None] * xa)
    hess.diagonal()[:d] += 1.0
    return hess


def logistic_hypergradient(
    w: torch.Tensor,
    b: torch.Tensor,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_p: torch.Tensor,
    y_p: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    C: float,
    outer_loss: Callable = mean_logistic_loss,
    damping: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form hypergradient of the attacker loss w.r.t. all poisoning points.

    ``(w, b)`` must be the optimum of the LR trained on ``D_tr U D_p``.

    Parameters
    ----------
    w, b : torch.Tensor
        Optimal parameters of the poisoned classifier.
    x_tr, y_tr : torch.Tensor
        Clean training data (labels in {0,1} or {-1,+1}).
    x_p, y_p : torch.Tensor
        Poisoning points (m, d) and their (fixed) labels.
    x_val, y_val : torch.Tensor
        Attacker's validation data. For targeted attacks pass the target labels.
    C : float
        Inverse regularization strength of the LR.
    outer_loss : callable
        ``outer_loss(scores, y_val_pm1) -> scalar``. Default: mean logistic loss.
    damping : float
        Optional Tikhonov damping added to H (H + damping * I).

    Returns
    -------
    (grad, A) : tuple
        ``grad`` has the shape of ``x_p`` and is ``dA/dx_p``; ``A`` is the
        attacker loss at ``(w, b)``.
    """
    shape = x_p.shape
    x_tr, x_p, x_val = (t.reshape(t.shape[0], -1).to(torch.float64) for t in (x_tr, x_p, x_val))
    y_tr, y_p, y_val = to_pm1(y_tr), to_pm1(y_p), to_pm1(y_val)
    d = x_tr.shape[1]

    loss, g_theta = outer_gradient(w, b, x_val, y_val, outer_loss)

    hess = logistic_hessian(w, b, torch.cat([x_tr, x_p]), torch.cat([y_tr, y_p]), C)
    if damping > 0:
        hess.diagonal().add_(damping)
    v = torch.linalg.solve(hess, g_theta)
    v_w, v_b = v[:d], v[d]

    p = torch.sigmoid(y_p * (x_p @ w + b))
    z = p * (1 - p)
    grad = C * ((y_p * (1 - p))[:, None] * v_w - (z * (x_p @ v_w + v_b))[:, None] * w)
    return grad.reshape(shape), loss


def implicit_hypergradient(
    inner_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    outer_loss: Callable[[torch.Tensor], torch.Tensor],
    theta: torch.Tensor,
    x_p: torch.Tensor,
    damping: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generic hypergradient via autograd (exact Hessian, any smooth learner).

    Parameters
    ----------
    inner_loss : callable
        ``inner_loss(theta, x_p) -> scalar``: training objective on
        ``D_tr U D_p`` as a function of flat parameters and poisoning points.
    outer_loss : callable
        ``outer_loss(theta) -> scalar``: attacker objective.
    theta : torch.Tensor
        Flat optimal parameters of the inner problem (1-D).
    x_p : torch.Tensor
        Poisoning points.

    Returns
    -------
    (grad, A) : tuple
        ``dA/dx_p`` and the outer loss value.

    Notes
    -----
    Forms the full (p x p) Hessian, so it is meant for small models (linear
    learners, verification of closed-form expressions). For large models
    replace ``torch.linalg.solve`` with conjugate gradient on Hessian-vector
    products.
    """
    theta = theta.detach().requires_grad_()
    a = outer_loss(theta)
    (g_out,) = torch.autograd.grad(a, theta)

    xp_const = x_p.detach()
    hess = torch.func.hessian(lambda t: inner_loss(t, xp_const))(theta.detach())
    if damping > 0:
        hess.diagonal().add_(damping)
    v = torch.linalg.solve(hess, g_out)

    xp = x_p.detach().clone().requires_grad_()
    (g_in,) = torch.autograd.grad(inner_loss(theta, xp), theta, create_graph=True)
    (grad,) = torch.autograd.grad(g_in @ v, xp)
    return -grad, a.detach()


# --------------------------------------------------------------------- SVM
def mean_hinge_loss(scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Average hinge loss ``max(0, 1 - y f)`` (labels in {-1, +1})."""
    return torch.relu(1 - y * scores).mean()


def svm_hypergradient(
    solution,
    kernel: Callable,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_p: torch.Tensor,
    y_p: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    C: float,
    outer_loss: Callable = mean_hinge_loss,
    jitter: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Hypergradient of the attacker loss for a (kernel) SVM via its KKT system.

    The hinge loss is not twice differentiable, so we do *not* differentiate
    the training objective. Instead, for a fixed partition of the training set
    into margin SVs ``S`` (0 < alpha < C), error vectors ``E`` (alpha = C) and
    reserve points ``R`` (alpha = 0), the SVM solution is characterized by the
    smooth equations (KKT conditions)

    .. math::

        F_i(u, X) = \sum_j Q_{ij}(X)\,\alpha_j + y_i b - 1 = 0 \quad (i\in S),
        \qquad F_0(u) = \sum_j y_j\alpha_j = 0,

    with unknowns ``u = (alpha_S, b)`` (alpha_E = C and alpha_R = 0 are
    constants). The implicit function theorem gives
    ``du/dx_c = -J^{-1} dF/dx_c`` with ``J = [[Q_SS, y_S], [y_S^T, 0]]``,
    so with ``lambda = J^{-T} dA/du``:

    .. math::

        \frac{dA}{dx_c} = \frac{\partial A}{\partial x_c}
                        - \lambda^\top \frac{\partial F}{\partial x_c}.

    For a poisoning point that is not a margin SV this reduces to Biggio et
    al. (ICML 2012):
    ``alpha_c y_c [ sum_k dl/df_k grad_{x_c} K(x_k, x_c)
    - sum_{i in S} lambda_i y_i grad_{x_c} K(x_i, x_c) ]``.
    Here all partial derivatives are taken with autograd, so any
    differentiable kernel works, and the case ``x_c in S`` is also covered.

    The result is the exact gradient wherever the partition (S, E, R) is
    locally constant; it is zero for points with alpha_c = 0.

    Parameters
    ----------
    solution : SVMDualSolution
        Solution of the SVM trained on ``D_tr U D_p`` (points in this order).
    kernel : callable
        Kernel function ``kernel(A, B) -> (len(A), len(B))``.
    jitter : float
        Same relative ridge used by the solver.

    Returns
    -------
    (grad, A) : tuple
        ``dA/dx_p`` (shape of ``x_p``) and the attacker loss.
    """
    shape = x_p.shape
    x_tr, x_p, x_val = (t.reshape(t.shape[0], -1).to(torch.float64) for t in (x_tr, x_p, x_val))
    y = torch.cat([to_pm1(y_tr), to_pm1(y_p)])
    y_val = to_pm1(y_val)

    S, E = solution.margin, solution.error
    s_idx = S.nonzero().squeeze(1)
    m = s_idx.numel()
    alpha_const = solution.alpha.detach().clone()
    alpha_const[S] = 0.0

    xp = x_p.detach().clone().requires_grad_()
    u = torch.cat([solution.alpha[S], solution.b.reshape(1)]).detach().requires_grad_()

    X = torch.cat([x_tr, xp])
    alpha = alpha_const.index_put((s_idx,), u[:m])
    b = u[m]
    coef = alpha * y
    with torch.no_grad():
        eps = jitter * kernel(x_tr, x_tr).diagonal().abs().max().clamp_min(1e-12)
        eps = max(eps.item(), (jitter * kernel(x_p, x_p).diagonal().abs().max()).item())

    # outer objective
    a = outer_loss(kernel(x_val, X) @ coef + b, y_val)
    if m == 0:
        # no margin SVs: b is not determined by equalities (it is the midpoint
        # of an interval) and alpha is constant -> only the direct term
        (grad,) = torch.autograd.grad(a, xp)
        return grad.reshape(shape), a.detach()

    # KKT residuals on the margin SVs (+ equality constraint)
    f_s = kernel(X[s_idx], X) @ coef + b
    F = torch.cat([y[s_idx] * f_s + eps * alpha[s_idx] - 1, (y * alpha).sum().reshape(1)])

    (da_du,) = torch.autograd.grad(a, u, retain_graph=True)
    with torch.no_grad():
        K_ss = kernel(X[s_idx], X[s_idx])
        J = torch.zeros(m + 1, m + 1, dtype=torch.float64)
        J[:m, :m] = y[s_idx, None] * y[None, s_idx] * K_ss + eps * torch.eye(m, dtype=torch.float64)
        J[:m, m] = y[s_idx]
        J[m, :m] = y[s_idx]
        lam = torch.linalg.solve(J.T, da_du)
    (grad,) = torch.autograd.grad(a - lam @ F, xp)
    return grad.reshape(shape), a.detach()
