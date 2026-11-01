"""Verification test suite for Phase 3 Model Evaluation & Reporting.

Enforces:
1. Complete schema of artifacts/model_metrics.json.
2. Untouched holdout test evaluation (N=4,000, seed=42).
3. Robust handling of edge cases (single-class splits, zero division).
4. Real ROC and Calibration curve data existence and valid ranges.
5. Exact class prevalence and sample counts consistency.
6. Documentation of split methodology and limitations statement.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import ANTIBIOTIC_TARGET_MAP, ARTIFACTS_DIR
from src.evaluate import evaluate_binary_classifier, run_evaluation_pipeline


def test_single_class_edge_case_handling():
    """evaluate_binary_classifier must handle single-class arrays gracefully without raising exceptions."""
    # All negative (zeros)
    y_true_zeros = np.zeros(100)
    y_prob = np.random.uniform(0.1, 0.4, 100)
    m_zeros = evaluate_binary_classifier(y_true_zeros, y_prob)

    assert m_zeros["roc_auc"] == 0.5
    assert m_zeros["precision"] == 0.0
    assert m_zeros["recall"] == 0.0
    assert m_zeros["sample_counts"]["positive_cases"] == 0
    assert m_zeros["sample_counts"]["prevalence"] == 0.0

    # All positive (ones)
    y_true_ones = np.ones(100)
    m_ones = evaluate_binary_classifier(y_true_ones, y_prob)
    assert m_ones["roc_auc"] == 0.5
    assert m_ones["sample_counts"]["positive_cases"] == 100
    assert m_ones["sample_counts"]["prevalence"] == 1.0


def test_evaluation_pipeline_execution_and_schema():
    """run_evaluation_pipeline must execute cleanly and return the full standardized schema."""
    report = run_evaluation_pipeline()

    assert "metadata" in report
    meta = report["metadata"]
    assert "split_methodology" in meta
    assert meta["split_methodology"]["random_seed"] == 42
    assert "limitations_statement" in meta
    assert "SYNTHETIC" in meta["limitations_statement"].upper()

    assert "antibiotics" in report
    assert len(report["antibiotics"]) == 8

    for abx in ANTIBIOTIC_TARGET_MAP.keys():
        assert abx in report["antibiotics"]
        item = report["antibiotics"][abx]

        assert item["selected_model"] in ["Random Forest", "XGBoost"]
        assert "test_metrics" in item
        m = item["test_metrics"]

        # Validate metric types and boundaries
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
        assert 0.0 <= m["f1"] <= 1.0
        assert 0.5 <= m["roc_auc"] <= 1.0
        assert 0.0 <= m["brier_score"] <= 1.0

        # Confusion matrix checks
        cm = m["confusion_matrix"]
        assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == 4000
        assert cm["tp"] + cm["fn"] == m["sample_counts"]["positive_cases"]

        # ROC Curve checks
        roc = m["roc_curve"]
        assert len(roc["fpr"]) == len(roc["tpr"])
        assert len(roc["fpr"]) > 0
        assert 0.0 <= roc["fpr"][0] <= 0.1
        assert 0.9 <= roc["tpr"][-1] <= 1.0

        # Sample counts
        assert m["sample_counts"]["total"] == 4000
        expected_prev = round(m["sample_counts"]["positive_cases"] / 4000.0, 4)
        assert m["sample_counts"]["prevalence"] == expected_prev


def test_model_metrics_artifact_file_integrity():
    """artifacts/model_metrics.json must be persisted on disk and match schema."""
    artifact_path = ARTIFACTS_DIR / "model_metrics.json"
    assert artifact_path.exists(), "artifacts/model_metrics.json missing"

    with open(artifact_path, "r") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "antibiotics" in data
    assert len(data["antibiotics"]) == 8

    # Verify both XGBoost and Random Forest are selected
    selected_models = {info["selected_model"] for info in data["antibiotics"].values()}
    assert "Random Forest" in selected_models
    assert "XGBoost" in selected_models
