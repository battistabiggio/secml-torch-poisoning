import pytest
import torch
from sklearn.linear_model import LogisticRegression as SkLR

from secmlt_poisoning import (
    BilevelPoisoningLogisticRegression,
    LogisticRegression,
    LogisticRegressionTrainer,
    fit_logistic_regression,
    logistic_hypergradient,
)
from secmlt_poisoning.hypergradient import mean_logistic_loss

torch.set_default_dtype(torch.float64)


def blobs(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (n,), generator=g)
    x = torch.randn(n, d, generator=g) + (2.0 * y[:, None] - 1) * 0.8
    return x, y


@pytest.fixture
def data():
    x_tr, y_tr = blobs(40, 5, 0)
    x_val, y_val = blobs(60, 5, 1)
    x_p, y_p = blobs(3, 5, 2)
    return x_tr, y_tr, x_val, y_val, x_p, 1 - y_p


@pytest.mark.parametrize("C", [0.1, 1.0, 10.0])
def test_fit_matches_sklearn(data, C):
    x, y = data[0], data[1]
    w, b = fit_logistic_regression(x, y, C=C)
    sk = SkLR(C=C, tol=1e-12, max_iter=10000).fit(x.numpy(), y.numpy())
    # sklearn (lbfgs) does not regularize the intercept either
    torch.testing.assert_close(w, torch.tensor(sk.coef_[0]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(b, torch.tensor(sk.intercept_[0]), atol=1e-5, rtol=1e-5)


def _outer_value(x_tr, y_tr, x_p, y_p, x_val, y_val, C):
    w, b = fit_logistic_regression(torch.cat([x_tr, x_p]), torch.cat([y_tr, y_p]), C=C, tol=1e-13)
    return mean_logistic_loss(x_val @ w + b, 2.0 * y_val - 1)


@pytest.mark.parametrize("C", [0.1, 1.0, 10.0])
def test_hypergradient_vs_finite_differences(data, C):
    x_tr, y_tr, x_val, y_val, x_p, y_p = data
    w, b = fit_logistic_regression(torch.cat([x_tr, x_p]), torch.cat([y_tr, y_p]), C=C, tol=1e-13)
    grad, _ = logistic_hypergradient(w, b, x_tr, y_tr, x_p, y_p, x_val, y_val, C)

    eps = 1e-4
    fd = torch.zeros_like(x_p)
    for i in range(x_p.shape[0]):
        for j in range(x_p.shape[1]):
            e = torch.zeros_like(x_p)
            e[i, j] = eps
            fd[i, j] = (
                _outer_value(x_tr, y_tr, x_p + e, y_p, x_val, y_val, C)
                - _outer_value(x_tr, y_tr, x_p - e, y_p, x_val, y_val, C)
            ) / (2 * eps)
    torch.testing.assert_close(grad, fd, atol=1e-6, rtol=1e-4)


def test_closed_form_equals_autograd(data):
    x_tr, y_tr, x_val, y_val, x_p, y_p = data
    kw = dict(C=1.0)
    a1 = BilevelPoisoningLogisticRegression(hypergradient="closed_form", **kw)
    a2 = BilevelPoisoningLogisticRegression(hypergradient="autograd", **kw)
    g1, o1 = a1.gradient(x_tr, y_tr, x_p, y_p, x_val, y_val)
    g2, o2 = a2.gradient(x_tr, y_tr, x_p, y_p, x_val, y_val)
    torch.testing.assert_close(g1, g2)
    torch.testing.assert_close(o1, o2)


def test_attack_increases_validation_loss(data):
    x_tr, y_tr, x_val, y_val, _, _ = data
    lo, hi = x_tr.min().item() - 1, x_tr.max().item() + 1
    attack = BilevelPoisoningLogisticRegression(
        C=1.0, lb=lo, ub=hi, step_size=0.5, max_iter=30, random_seed=0
    )
    res = attack.run(x_tr, y_tr, x_val, y_val, n_points=4)
    h = res.history["objective"]
    assert h[-1] > h[0]
    assert all(b >= a for a, b in zip(h, h[1:]))  # monotone with line search
    assert res.x_poison.min() >= lo and res.x_poison.max() <= hi


def test_targeted_attack_decreases_target_loss(data):
    x_tr, y_tr, x_val, y_val, _, _ = data
    attack = BilevelPoisoningLogisticRegression(
        C=1.0, lb=-5, ub=5, step_size=0.5, max_iter=20, y_target=1, random_seed=0
    )
    res = attack.run(x_tr, y_tr, x_val, y_val, n_points=4)
    assert res.history["objective"][-1] < res.history["objective"][0]


def test_secmlt_integration(data):
    from secmlt.metrics.classification import Accuracy
    from secmlt.models.pytorch.base_pytorch_nn import BasePyTorchClassifier
    from torch.utils.data import DataLoader, TensorDataset

    x_tr, y_tr, x_val, y_val, _, _ = data
    clf = BasePyTorchClassifier(LogisticRegression(5, C=1.0), trainer=LogisticRegressionTrainer())
    clf.train(DataLoader(TensorDataset(x_tr, y_tr), batch_size=8))
    acc = Accuracy()(clf, DataLoader(TensorDataset(x_val, y_val), batch_size=16))
    w, b = fit_logistic_regression(x_tr, y_tr, C=1.0)
    ref = ((x_val @ w + b > 0).long() == y_val).double().mean()
    assert abs(acc.item() - ref.item()) < 1e-9
