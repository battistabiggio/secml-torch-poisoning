"""Bilevel poisoning attacks for secml-torch."""

from secmlt_poisoning.attacks.base import BilevelPoisoningAttack, PoisoningResult, poisoned_dataset
from secmlt_poisoning.attacks.bilevel_logistic import BilevelPoisoningLogisticRegression
from secmlt_poisoning.attacks.bilevel_svm import BilevelPoisoningSVM
from secmlt_poisoning.hypergradient import (
    implicit_hypergradient,
    logistic_hypergradient,
    mean_hinge_loss,
    mean_logistic_loss,
    svm_hypergradient,
)
from secmlt_poisoning.models.logistic_regression import (
    LogisticRegression,
    LogisticRegressionTrainer,
    fit_logistic_regression,
)
from secmlt_poisoning.models.svm import SVM, LinearKernel, RBFKernel, SVMTrainer, solve_svm_dual

__all__ = [
    "SVM",
    "BilevelPoisoningAttack",
    "BilevelPoisoningLogisticRegression",
    "BilevelPoisoningSVM",
    "LinearKernel",
    "LogisticRegression",
    "LogisticRegressionTrainer",
    "PoisoningResult",
    "RBFKernel",
    "SVMTrainer",
    "fit_logistic_regression",
    "implicit_hypergradient",
    "logistic_hypergradient",
    "mean_hinge_loss",
    "mean_logistic_loss",
    "poisoned_dataset",
    "solve_svm_dual",
    "svm_hypergradient",
]
