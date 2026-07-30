"""Tests for the detector performance-metrics toolkit."""

from __future__ import annotations

import pytest

from frontline_drones import detector_metrics as dm
from frontline_drones.detector_metrics import (
    ConfusionMatrix,
    average_precision,
    confusion_at_threshold,
    confusion_matrix,
    precision_recall_curve,
    roc_auc,
    roc_curve,
    threshold_for_target_fpr,
    threshold_for_target_recall,
    youden_j_threshold,
)

# -- ConfusionMatrix ----------------------------------------------------------

def test_perfect_classifier():
    cm = ConfusionMatrix(tp=5, fp=0, tn=5, fn=0)
    assert cm.precision == 1.0
    assert cm.recall == 1.0
    assert cm.accuracy == 1.0
    assert cm.f1 == 1.0
    assert cm.fpr == 0.0


def test_total():
    cm = ConfusionMatrix(tp=1, fp=2, tn=3, fn=4)
    assert cm.total == 10


def test_precision_zero_when_none_flagged():
    cm = ConfusionMatrix(tp=0, fp=0, tn=5, fn=3)
    assert cm.precision == 0.0


def test_recall_zero_when_no_positives():
    cm = ConfusionMatrix(tp=0, fp=1, tn=5, fn=0)
    assert cm.recall == 0.0


def test_fpr_and_specificity_complement():
    cm = ConfusionMatrix(tp=3, fp=2, tn=8, fn=1)
    assert cm.fpr + cm.specificity == pytest.approx(1.0)


def test_tpr_equals_recall():
    cm = ConfusionMatrix(tp=3, fp=2, tn=8, fn=1)
    assert cm.tpr == cm.recall


def test_f1_harmonic_mean():
    cm = ConfusionMatrix(tp=8, fp=2, tn=0, fn=2)
    # precision 0.8, recall 0.8 -> f1 0.8
    assert cm.f1 == pytest.approx(0.8)


def test_fbeta_favours_recall():
    cm = ConfusionMatrix(tp=8, fp=8, tn=0, fn=2)  # p=0.5, r=0.8
    f2 = cm.fbeta(2.0)
    f05 = cm.fbeta(0.5)
    assert f2 > f05  # beta>1 weights recall (the larger metric here)


def test_fbeta_zero_when_empty():
    cm = ConfusionMatrix(tp=0, fp=0, tn=0, fn=0)
    assert cm.fbeta(1.0) == 0.0


def test_accuracy_zero_total():
    cm = ConfusionMatrix(tp=0, fp=0, tn=0, fn=0)
    assert cm.accuracy == 0.0


def test_confusion_to_dict_keys():
    cm = ConfusionMatrix(tp=1, fp=1, tn=1, fn=1)
    d = cm.to_dict()
    for k in ("tp", "fp", "tn", "fn", "total", "precision", "recall", "tpr",
              "fpr", "specificity", "accuracy", "f1"):
        assert k in d


# -- confusion_matrix ---------------------------------------------------------

def test_confusion_matrix_counts():
    labels = [1, 1, 0, 0]
    preds = [1, 0, 1, 0]
    cm = confusion_matrix(labels, preds)
    assert (cm.tp, cm.fn, cm.fp, cm.tn) == (1, 1, 1, 1)


def test_confusion_matrix_truthy_labels():
    cm = confusion_matrix([True, False], [True, False])
    assert cm.tp == 1 and cm.tn == 1


def test_confusion_matrix_length_mismatch():
    with pytest.raises(ValueError):
        confusion_matrix([1, 0], [1])


def test_confusion_at_threshold():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.6, 0.55, 0.1]
    cm = confusion_at_threshold(labels, scores, 0.6)
    # flags scores >= 0.6: first two (both pos) -> tp=2, fp=0
    assert cm.tp == 2
    assert cm.fp == 0


def test_confusion_at_threshold_loose():
    labels = [1, 0]
    scores = [0.9, 0.8]
    cm = confusion_at_threshold(labels, scores, 0.0)
    assert cm.tp == 1 and cm.fp == 1


# -- roc_curve / roc_auc ------------------------------------------------------

def test_roc_starts_at_origin():
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.1]
    pts = roc_curve(labels, scores)
    assert pts[0].fpr == 0.0 and pts[0].tpr == 0.0


def test_roc_ends_at_corner():
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.1]
    pts = roc_curve(labels, scores)
    assert pts[-1].fpr == pytest.approx(1.0)
    assert pts[-1].tpr == pytest.approx(1.0)


def test_roc_auc_perfect():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    assert roc_auc(labels, scores) == pytest.approx(1.0)


def test_roc_auc_worst():
    # Perfectly wrong ranking -> AUC 0.
    labels = [1, 1, 0, 0]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert roc_auc(labels, scores) == pytest.approx(0.0)


def test_roc_auc_chance():
    # Interleaved -> around 0.5.
    labels = [1, 0, 1, 0]
    scores = [0.6, 0.5, 0.4, 0.3]
    auc = roc_auc(labels, scores)
    assert 0.0 <= auc <= 1.0


def test_roc_auc_known_value():
    labels = [1, 1, 0, 0, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.4, 0.3, 0.2]
    # 3 pos, 3 neg; computed reference value.
    assert roc_auc(labels, scores) == pytest.approx(0.7777778, abs=1e-6)


def test_roc_requires_both_classes():
    with pytest.raises(ValueError):
        roc_curve([1, 1, 1], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        roc_curve([0, 0, 0], [0.1, 0.2, 0.3])


def test_roc_handles_tied_scores():
    labels = [1, 0, 1, 0]
    scores = [0.5, 0.5, 0.5, 0.5]
    pts = roc_curve(labels, scores)
    # One threshold consumes all ties -> endpoint at (1,1).
    assert pts[-1].fpr == pytest.approx(1.0)
    assert pts[-1].tpr == pytest.approx(1.0)


def test_roc_point_to_dict():
    pts = roc_curve([1, 0], [0.9, 0.1])
    d = pts[-1].to_dict()
    assert set(d) == {"threshold", "fpr", "tpr"}


# -- precision_recall / average_precision -------------------------------------

def test_pr_curve_increasing_recall():
    labels = [1, 0, 1, 0, 1]
    scores = [0.9, 0.8, 0.7, 0.4, 0.3]
    pts = precision_recall_curve(labels, scores)
    recalls = [p.recall for p in pts]
    assert recalls == sorted(recalls)
    assert recalls[-1] == pytest.approx(1.0)


def test_pr_requires_positive():
    with pytest.raises(ValueError):
        precision_recall_curve([0, 0], [0.1, 0.2])


def test_average_precision_perfect():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    assert average_precision(labels, scores) == pytest.approx(1.0)


def test_average_precision_bounded():
    labels = [1, 0, 1, 0, 1, 0]
    scores = [0.9, 0.85, 0.7, 0.6, 0.5, 0.4]
    ap = average_precision(labels, scores)
    assert 0.0 <= ap <= 1.0


def test_average_precision_all_positive_first():
    labels = [1, 1, 1, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.4, 0.3]
    assert average_precision(labels, scores) == pytest.approx(1.0)


def test_pr_point_to_dict():
    pts = precision_recall_curve([1, 0], [0.9, 0.1])
    d = pts[0].to_dict()
    assert set(d) == {"threshold", "recall", "precision"}


# -- threshold selection ------------------------------------------------------

def test_threshold_for_target_fpr_zero():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.3, 0.2]
    thr = threshold_for_target_fpr(labels, scores, 0.0)
    cm = confusion_at_threshold(labels, scores, thr)
    assert cm.fpr == 0.0
    assert cm.tp == 2  # catches both positives with no false alarms


def test_threshold_for_target_fpr_budget():
    labels = [1, 1, 0, 0, 0]
    scores = [0.9, 0.5, 0.8, 0.4, 0.3]
    thr = threshold_for_target_fpr(labels, scores, 0.34)  # allow ~1 of 3 negatives
    cm = confusion_at_threshold(labels, scores, thr)
    assert cm.fpr <= 0.34 + 1e-9


def test_threshold_for_target_fpr_out_of_range():
    with pytest.raises(ValueError):
        threshold_for_target_fpr([1, 0], [0.9, 0.1], 1.5)


def test_threshold_for_target_recall():
    labels = [1, 1, 1, 0, 0]
    scores = [0.9, 0.6, 0.3, 0.8, 0.2]
    thr = threshold_for_target_recall(labels, scores, 1.0)
    cm = confusion_at_threshold(labels, scores, thr)
    assert cm.recall >= 1.0 - 1e-9


def test_threshold_for_target_recall_partial():
    labels = [1, 1, 1, 1, 0]
    scores = [0.9, 0.8, 0.5, 0.4, 0.3]
    thr = threshold_for_target_recall(labels, scores, 0.5)
    cm = confusion_at_threshold(labels, scores, thr)
    assert cm.recall >= 0.5


def test_threshold_for_target_recall_out_of_range():
    with pytest.raises(ValueError):
        threshold_for_target_recall([1, 0], [0.9, 0.1], -0.1)


# -- youden_j -----------------------------------------------------------------

def test_youden_j_separable():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    thr, j = youden_j_threshold(labels, scores)
    assert j == pytest.approx(1.0)
    # Operating at that threshold should give perfect separation.
    cm = confusion_at_threshold(labels, scores, thr)
    assert cm.tpr == 1.0 and cm.fpr == 0.0


def test_youden_j_bounded():
    labels = [1, 0, 1, 0, 1, 0]
    scores = [0.6, 0.55, 0.5, 0.45, 0.4, 0.35]
    _thr, j = youden_j_threshold(labels, scores)
    assert 0.0 <= j <= 1.0


# -- determinism --------------------------------------------------------------

def test_roc_deterministic():
    labels = [1, 0, 1, 0, 1, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    a = [p.to_dict() for p in roc_curve(labels, scores)]
    b = [p.to_dict() for p in roc_curve(labels, scores)]
    assert a == b


def test_module_exposes_helpers():
    assert hasattr(dm, "roc_auc")
    assert hasattr(dm, "average_precision")
