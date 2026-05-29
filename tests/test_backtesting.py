"""Tests for the backtesting framework metric calculations."""
import numpy as np
from src.backtesting.backtester import (
    _confusion,
    _metrics_from_confusion,
    evaluate_at_threshold,
    threshold_sweep,
    find_best_threshold,
)


def test_confusion_perfect():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    c = _confusion(y_true, y_pred)
    assert c == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}


def test_confusion_all_wrong():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1])
    c = _confusion(y_true, y_pred)
    assert c == {"tp": 0, "fp": 2, "tn": 0, "fn": 2}


def test_metrics_perfect():
    c = {"tp": 10, "fp": 0, "tn": 10, "fn": 0}
    m = _metrics_from_confusion(c)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["accuracy"] == 1.0


def test_metrics_zero_division_safe():
    """No positive predictions — precision should be 0, not crash."""
    c = {"tp": 0, "fp": 0, "tn": 10, "fn": 5}
    m = _metrics_from_confusion(c)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_metrics_half():
    c = {"tp": 5, "fp": 5, "tn": 5, "fn": 5}
    m = _metrics_from_confusion(c)
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5
    assert m["accuracy"] == 0.5


def test_evaluate_at_threshold():
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    labels = np.array([0, 0, 1, 1])
    m = evaluate_at_threshold(scores, labels, 0.5)
    # scores >= 0.5: indices 2,3 -> both illicit -> perfect
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["threshold"] == 0.5


def test_threshold_sweep_length():
    scores = np.random.rand(100)
    labels = (np.random.rand(100) > 0.5).astype(int)
    sweep = threshold_sweep(scores, labels, steps=19)
    assert len(sweep) == 19
    for m in sweep:
        assert "threshold" in m
        assert "f1" in m


def test_find_best_threshold_picks_max_f1():
    sweep = [
        {"threshold": 0.3, "f1": 0.2},
        {"threshold": 0.5, "f1": 0.8},
        {"threshold": 0.7, "f1": 0.5},
    ]
    best = find_best_threshold(sweep)
    assert best["threshold"] == 0.5
    assert best["f1"] == 0.8


def test_higher_threshold_fewer_positives():
    """Higher threshold should never increase positive predictions."""
    scores = np.linspace(0, 1, 100)
    labels = (scores > 0.5).astype(int)
    low = evaluate_at_threshold(scores, labels, 0.2)
    high = evaluate_at_threshold(scores, labels, 0.8)
    low_pos = low["confusion"]["tp"] + low["confusion"]["fp"]
    high_pos = high["confusion"]["tp"] + high["confusion"]["fp"]
    assert high_pos <= low_pos
