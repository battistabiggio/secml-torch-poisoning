import pytest
import torch
from sklearn.svm import SVC

from secmlt_poisoning import BilevelPoisoningSVM, LinearKernel, RBFKernel, SVM, SVMTrainer
from secmlt_poisoning.hypergradient import mean_hinge_loss, mean_logistic_loss

torch.set_default_dtype(torch.float64)


def blobs(n, d, seed, sep=1.0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (n,), generator=g)
    x = torch.randn(n, d, generator=g) + (2.0 * y[:, None] - 1) * sep
    return x, y


KERNELS = [
    (LinearKernel(), {"kernel": "linear"}),
    (RBFKernel(0.5), {"kernel": "rbf", "gamma": 0.5}),
    (RBFKernel(5.0), {"kernel": "rbf", "gamma": 5.0}),
]


@pytest.mark.parametrize(("kernel", "sk"), KERNELS)
@pytest.mark.parametrize("C", [0.01, 1.0, 100.0])
def test_solver_matches_libsvm(kernel, sk, C):
    x, y = blobs(100, 2, 0)
    m = SVM(C, kernel).fit(x, y)
    assert m.solution.converged
    s = SVC(C=C, tol=1e-10, **sk).fit(x.numpy(), y.numpy())
    f_sk = torch.tensor(s.decision_function(x.numpy()))
    torch.testing.assert_close(m.score(x), f_sk, atol=1e-3 * max(1.0, f_sk.abs().max().item()), rtol=0)
    # warm start after a small perturbation converges to the same solution
    x2 = x.clone()
    x2[0] += 0.05
    m_warm = SVM(C, kernel).fit(x2, y, alpha0=m.solution.alpha)
    m_cold = SVM(C, kernel).fit(x2, y)
    torch.testing.assert_close(m_warm.score(x2), m_cold.score(x2), atol=1e-7, rtol=0)


def test_duplicated_points_and_high_dim():
    x, y = blobs(30, 50, 1)
    x = torch.cat([x, x[:5]])  # duplicates -> singular kernel matrix
    y = torch.cat([y, y[:5]])
    m = SVM(1.0).fit(x, y)
    assert m.solution.converged


@pytest.mark.parametrize("kernel", [RBFKernel(0.5), LinearKernel()])
@pytest.mark.parametrize("outer", [mean_hinge_loss, mean_logistic_loss])
@pytest.mark.parametrize("C", [0.1, 1.0, 10.0])
def test_hypergradient_vs_finite_differences(kernel, outer, C):
    # d=10 > |S| for the linear kernel, so the hyperplane is not pinned by S
    d = 2 if isinstance(kernel, RBFKernel) else 10
    x_tr, y_tr = blobs(40, d, 0)
    x_val, y_val = blobs(80, d, 1)
    x_p, y_p = x_tr[:3] + 0.05, 1 - y_tr[:3]
    attack = BilevelPoisoningSVM(C=C, kernel=kernel, outer_loss=outer)
    g, _ = attack.gradient(x_tr, y_tr, x_p, y_p, x_val, y_val)

    def obj(z):
        attack._warm = None
        return attack.objective(x_tr, y_tr, z, y_p, x_val, y_val)

    eps = 1e-6
    fd = torch.zeros_like(x_p)
    for i in range(x_p.shape[0]):
        for j in range(x_p.shape[1]):
            e = torch.zeros_like(x_p)
            e[i, j] = eps
            fd[i, j] = (obj(x_p + e) - obj(x_p - e)) / (2 * eps)
    assert g.abs().max() > 1e-4  # non-trivial gradient
    torch.testing.assert_close(g, fd, atol=1e-7, rtol=1e-4)


def test_margin_sv_poisoning_point():
    """Gradient is also exact when the poisoning point is itself a margin SV."""
    x_tr, y_tr = blobs(40, 2, 3)
    x_val, y_val = blobs(80, 2, 4)
    attack = BilevelPoisoningSVM(C=1.0, kernel=RBFKernel(0.5))
    # search a location where the poisoning point lands on the margin
    for x0 in torch.linspace(-2, 2, 81):
        x_p = torch.tensor([[x0.item(), 0.3]])
        y_p = torch.tensor([1])
        sol = attack.train(x_tr, y_tr, x_p, y_p, warm_start=False).solution
        if sol.margin[-1]:
            break
    else:
        pytest.skip("no margin-SV configuration found")
    g, _ = attack.gradient(x_tr, y_tr, x_p, y_p, x_val, y_val)
    eps, fd = 1e-7, torch.zeros_like(x_p)
    for j in range(2):
        e = torch.zeros_like(x_p)
        e[0, j] = eps
        attack._warm = None
        a = attack.objective(x_tr, y_tr, x_p + e, y_p, x_val, y_val)
        attack._warm = None
        b = attack.objective(x_tr, y_tr, x_p - e, y_p, x_val, y_val)
        fd[0, j] = (a - b) / (2 * eps)
    torch.testing.assert_close(g, fd, atol=1e-6, rtol=1e-3)


def test_attack_increases_validation_loss():
    x_tr, y_tr = blobs(60, 2, 0)
    x_val, y_val = blobs(100, 2, 1)
    attack = BilevelPoisoningSVM(C=1.0, kernel=RBFKernel(0.5), lb=-4, ub=4, step_size=0.5,
                                 max_iter=30, random_seed=0)
    res = attack.run(x_tr, y_tr, x_val, y_val, n_points=5)
    h = res.history["objective"]
    assert h[-1] > h[0]
    assert all(b >= a for a, b in zip(h, h[1:]))
    assert res.x_poison.min() >= -4 and res.x_poison.max() <= 4


def test_secmlt_integration():
    from secmlt.metrics.classification import Accuracy
    from secmlt.models.pytorch.base_pytorch_nn import BasePyTorchClassifier
    from torch.utils.data import DataLoader, TensorDataset

    x_tr, y_tr = blobs(60, 2, 0)
    x_val, y_val = blobs(100, 2, 1)
    clf = BasePyTorchClassifier(SVM(1.0, RBFKernel(0.5)), trainer=SVMTrainer())
    clf.train(DataLoader(TensorDataset(x_tr, y_tr), batch_size=8))
    acc = Accuracy()(clf, DataLoader(TensorDataset(x_val, y_val), batch_size=16))
    ref = ((clf.model.score(x_val) > 0).long() == y_val).double().mean()
    assert abs(acc.item() - ref.item()) < 1e-9
