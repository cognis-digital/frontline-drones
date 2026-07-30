"""Detector performance metrics for counter-UAS evaluation (offline analysis).

Before a detection pipeline is trusted, its performance is measured against
labelled truth: how often it catches a real drone (recall / true-positive rate)
versus how often it cries wolf (false-alarm rate). This module is a small,
stdlib-only, deterministic toolkit for exactly that offline evaluation - confusion
matrices, precision/recall/F1, ROC and precision-recall curves with trapezoidal
AUC, and threshold selection for a target false-alarm or detection rate.

Scope: pure offline scoring math over labelled arrays. There is no live
detection, tasking or engagement content; this is how an analyst *audits* a
detector's quality. Labels are truthy/falsey (1 = real drone present, 0 = clutter);
scores are real-valued detector confidences where higher means "more likely a drone".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfusionMatrix:
    """Binary confusion matrix with the standard derived rates.

    Attributes:
        tp: True positives (real drone, flagged).
        fp: False positives (clutter, flagged) - the false alarms.
        tn: True negatives (clutter, not flagged).
        fn: False negatives (real drone, missed) - the misses.
    """

    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        """TP / (TP + FP): of everything flagged, how much was real. 0 if none flagged."""
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        """TP / (TP + FN): of all real drones, how many were caught (a.k.a. TPR)."""
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def tpr(self) -> float:
        """True-positive rate == recall."""
        return self.recall

    @property
    def fpr(self) -> float:
        """FP / (FP + TN): the false-alarm rate. 0 when there are no negatives."""
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0

    @property
    def specificity(self) -> float:
        """TN / (TN + FP): how much clutter was correctly ignored."""
        denom = self.tn + self.fp
        return self.tn / denom if denom else 0.0

    @property
    def accuracy(self) -> float:
        """(TP + TN) / total: fraction of all calls that were correct."""
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall; 0 when both are 0."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def fbeta(self, beta: float) -> float:
        """Weighted F-measure; ``beta`` > 1 favours recall, < 1 favours precision."""
        p, r = self.precision, self.recall
        b2 = beta * beta
        denom = b2 * p + r
        return (1 + b2) * p * r / denom if denom else 0.0

    def to_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "total": self.total,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "tpr": round(self.tpr, 6),
            "fpr": round(self.fpr, 6),
            "specificity": round(self.specificity, 6),
            "accuracy": round(self.accuracy, 6),
            "f1": round(self.f1, 6),
        }


def _to_bool(label) -> bool:
    """Interpret a label as presence (truthy) or absence (falsey)."""
    return bool(label)


def confusion_matrix(labels, predictions) -> ConfusionMatrix:
    """Tally a :class:`ConfusionMatrix` from truth ``labels`` and boolean ``predictions``.

    Both are equal-length iterables; each element is truthy (drone/flagged) or
    falsey (clutter/not-flagged). Raises ``ValueError`` on a length mismatch.
    """
    labels = list(labels)
    predictions = list(predictions)
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must be the same length")
    tp = fp = tn = fn = 0
    for truth, pred in zip(labels, predictions, strict=True):
        t, p = _to_bool(truth), _to_bool(pred)
        if t and p:
            tp += 1
        elif not t and p:
            fp += 1
        elif not t and not p:
            tn += 1
        else:
            fn += 1
    return ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)


def confusion_at_threshold(labels, scores, threshold: float) -> ConfusionMatrix:
    """Confusion matrix when flagging every sample whose score >= ``threshold``."""
    scores = list(scores)
    preds = [s >= threshold for s in scores]
    return confusion_matrix(labels, preds)


def _pair_and_sort(labels, scores) -> list[tuple[float, bool]]:
    labels = list(labels)
    scores = list(scores)
    if len(labels) != len(scores):
        raise ValueError("labels and scores must be the same length")
    pairs = [(float(s), _to_bool(lab)) for s, lab in zip(scores, labels, strict=True)]
    # Sort by descending score; ties keep positives first so a shared threshold
    # counts them deterministically.
    pairs.sort(key=lambda p: (-p[0], not p[1]))
    return pairs


@dataclass(frozen=True)
class RocPoint:
    """One point on a ROC curve."""

    threshold: float
    fpr: float
    tpr: float

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "fpr": round(self.fpr, 6),
            "tpr": round(self.tpr, 6),
        }


def roc_curve(labels, scores) -> list[RocPoint]:
    """Compute the ROC curve as a list of :class:`RocPoint`, low FPR to high.

    Sweeps the decision threshold from strictest to loosest. The curve begins at
    (fpr=0, tpr=0) with an infinite threshold and ends at (fpr=1, tpr=1). Raises
    ``ValueError`` if there are no positive or no negative labels (rate undefined).
    """
    pairs = _pair_and_sort(labels, scores)
    total_pos = sum(1 for _s, t in pairs if t)
    total_neg = len(pairs) - total_pos
    if total_pos == 0 or total_neg == 0:
        raise ValueError("ROC needs at least one positive and one negative label")

    points: list[RocPoint] = [RocPoint(threshold=float("inf"), fpr=0.0, tpr=0.0)]
    tp = fp = 0
    i = 0
    n = len(pairs)
    while i < n:
        thr = pairs[i][0]
        # Consume all samples sharing this score before recording a point.
        while i < n and pairs[i][0] == thr:
            if pairs[i][1]:
                tp += 1
            else:
                fp += 1
            i += 1
        points.append(RocPoint(threshold=thr, fpr=fp / total_neg, tpr=tp / total_pos))
    return points


def _trapezoid_auc(xs: list[float], ys: list[float]) -> float:
    area = 0.0
    for (x0, y0), (x1, y1) in zip(zip(xs, ys, strict=True), zip(xs[1:], ys[1:], strict=True), strict=False):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area


def roc_auc(labels, scores) -> float:
    """Area under the ROC curve via the trapezoid rule, in [0, 1].

    0.5 is chance; 1.0 is a perfect ranking of positives above negatives.
    """
    pts = roc_curve(labels, scores)
    xs = [p.fpr for p in pts]
    ys = [p.tpr for p in pts]
    return _trapezoid_auc(xs, ys)


@dataclass(frozen=True)
class PrPoint:
    """One point on a precision-recall curve."""

    threshold: float
    recall: float
    precision: float

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "recall": round(self.recall, 6),
            "precision": round(self.precision, 6),
        }


def precision_recall_curve(labels, scores) -> list[PrPoint]:
    """Precision-recall curve as a list of :class:`PrPoint`, increasing recall.

    Sweeps the threshold from strictest to loosest, one point per distinct score.
    Raises ``ValueError`` when there are no positive labels.
    """
    pairs = _pair_and_sort(labels, scores)
    total_pos = sum(1 for _s, t in pairs if t)
    if total_pos == 0:
        raise ValueError("precision-recall needs at least one positive label")

    points: list[PrPoint] = []
    tp = fp = 0
    i = 0
    n = len(pairs)
    while i < n:
        thr = pairs[i][0]
        while i < n and pairs[i][0] == thr:
            if pairs[i][1]:
                tp += 1
            else:
                fp += 1
            i += 1
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / total_pos
        points.append(PrPoint(threshold=thr, recall=recall, precision=precision))
    return points


def average_precision(labels, scores) -> float:
    """Average precision: recall-weighted mean precision (area under the PR curve).

    Computed as ``sum (R_k - R_{k-1}) * P_k`` over the PR points, the standard
    information-retrieval AP. Raises ``ValueError`` with no positive labels.
    """
    pts = precision_recall_curve(labels, scores)
    ap = 0.0
    prev_recall = 0.0
    for p in pts:
        ap += (p.recall - prev_recall) * p.precision
        prev_recall = p.recall
    return ap


def threshold_for_target_fpr(labels, scores, max_fpr: float) -> float:
    """Loosest threshold whose false-alarm rate does not exceed ``max_fpr``.

    Returns the score threshold that maximises detections while keeping FPR at or
    below the operational false-alarm budget. If even the strictest threshold
    exceeds ``max_fpr`` (e.g. ``max_fpr`` < 0), returns ``inf`` (flag nothing).
    """
    if not 0.0 <= max_fpr <= 1.0:
        raise ValueError("max_fpr must be in [0, 1]")
    pts = roc_curve(labels, scores)
    best_thr = float("inf")
    best_tpr = -1.0
    for p in pts:
        if p.fpr <= max_fpr and p.tpr >= best_tpr:
            best_tpr = p.tpr
            best_thr = p.threshold
    return best_thr


def threshold_for_target_recall(labels, scores, min_recall: float) -> float:
    """Strictest threshold that still achieves at least ``min_recall``.

    Returns the score threshold that meets the required detection rate while
    admitting the fewest false alarms. Raises ``ValueError`` for a recall target
    outside [0, 1] or when there are no positive labels.
    """
    if not 0.0 <= min_recall <= 1.0:
        raise ValueError("min_recall must be in [0, 1]")
    pts = precision_recall_curve(labels, scores)
    # Points are in increasing-recall order; the first meeting the target is the
    # strictest (highest) threshold that does so.
    for p in pts:
        if p.recall >= min_recall:
            return p.threshold
    return float("-inf")  # unreachable target: loosest threshold flags everything


def youden_j_threshold(labels, scores) -> tuple[float, float]:
    """Return ``(threshold, J)`` maximising Youden's J = TPR - FPR on the ROC curve.

    Youden's J picks the operating point that best separates drones from clutter
    when misses and false alarms are weighted equally. Returns the threshold and
    the achieved J in [0, 1].
    """
    pts = roc_curve(labels, scores)
    best = max(pts, key=lambda p: (p.tpr - p.fpr, p.threshold))
    return best.threshold, best.tpr - best.fpr
