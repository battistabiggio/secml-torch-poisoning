# secmlt-poisoning

Bilevel (gradient-based) poisoning attacks for [secml-torch](https://github.com/pralab/secml-torch),
ported from `secml.adv.attacks.poisoning`.

Currently implemented (binary classification):
- L2 logistic regression (`CAttackPoisoningLogisticRegression` in secml): closed-form hypergradient;
- SVM with linear / RBF / any differentiable kernel (`CAttackPoisoningSVM`, Biggio et al. ICML 2012):
  exact active-set dual solver + implicit differentiation of the KKT conditions of the margin SVs.

```
src/secmlt_poisoning/
  models/logistic_regression.py   LR nn.Module + exact Newton solver + secml-torch trainer
  hypergradient.py                closed-form LR hypergradient + generic autograd implicit diff
  models/svm.py                   kernel SVM nn.Module + exact active-set dual solver (warm-startable)
  hypergradient.py                (also) svm_hypergradient: implicit diff. of the KKT system
  attacks/base.py                 shared projected-gradient loop with line search
  attacks/bilevel_logistic.py     BilevelPoisoningLogisticRegression
  attacks/bilevel_svm.py          BilevelPoisoningSVM
tests/                            hypergradient vs finite differences / autograd, sklearn check, attack sanity
notebooks/poisoning_logistic_regression.ipynb   derivation, 2D examples, MNIST 4 vs 0
notebooks/poisoning_svm.ipynb                   KKT derivation, 2D (RBF + linear), MNIST 4 vs 0
```

```bash
python -m venv .venv --system-site-packages && .venv/bin/pip install -e ".[test,notebooks]"
.venv/bin/python -m pytest tests
```

```python
from secmlt_poisoning import BilevelPoisoningLogisticRegression
attack = BilevelPoisoningLogisticRegression(C=1.0, lb=0, ub=1, step_size=2.0, max_iter=200)
res = attack.run(x_tr, y_tr, x_val, y_val, n_points=10)   # labels in {0,1}
res.x_poison, res.y_poison, res.model, res.history["objective"]
```
