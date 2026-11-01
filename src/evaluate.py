"""Scientific Evaluation and Benchmarking Module for AMR-Predict.

Calculates comprehensive classification, discrimination, and calibration metrics
on the untouched final test split (N=4,000) across all 8 target antibiotics.
Provides standalone CLI execution via:
    python src/evaluate.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
import joblib

from src.config import (
    DATASET_PATH,
    MODELS_DIR,
    ARTIFACTS_DIR,
    ANTIBIOTIC_TARGET_MAP,
    MODEL_FEATURES,
    MEDICAL_DISCLAIMER,
)
from src.preprocessing import load_preprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("AMR_Evaluate")


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    downsample_curve_points: int = 50,
) -> Dict[str, Any]:
    """Calculates comprehensive classification & calibration metrics for a single model.

    Handles class imbalance, edge-case single-class partitions, and probability bounds gracefully.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probability of resistance (class 1).
        threshold: Decision threshold for discrete classification (default: 0.5).
        downsample_curve_points: Max points to keep for ROC/PR curves to optimize payload size.

    Returns:
        Structured metrics dictionary with core metrics, confusion matrix, curves, and prevalence.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(y_true) == 0:
        return {"error": "Empty evaluation array"}

    y_prob = np.clip(y_prob, 0.0, 1.0)
    y_pred = (y_prob >= threshold).astype(int)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    prevalence = float(n_pos / len(y_true))

    # Core metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # ROC-AUC with edge-case handling for single-class splits
    unique_classes = np.unique(y_true)
    if len(unique_classes) >= 2:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = 0.5
    else:
        logger.warning("Single-class evaluation split detected; defaulting ROC-AUC to 0.5")
        roc_auc = 0.5

    brier = float(brier_score_loss(y_true, y_prob))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ROC Curve
    if len(unique_classes) >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        if len(fpr) > downsample_curve_points:
            indices = np.linspace(0, len(fpr) - 1, downsample_curve_points, dtype=int)
            fpr = fpr[indices]
            tpr = tpr[indices]
    else:
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])

    # Precision-Recall Curve
    if len(unique_classes) >= 2:
        pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_prob)
        if len(pr_prec) > downsample_curve_points:
            indices = np.linspace(0, len(pr_prec) - 1, downsample_curve_points, dtype=int)
            pr_prec = pr_prec[indices]
            pr_rec = pr_rec[indices]
    else:
        pr_prec, pr_rec = np.array([prevalence, prevalence]), np.array([1.0, 0.0])

    # Calibration Curve (Reliability Diagram)
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        calib_data = {
            "prob_true": [round(float(x), 4) for x in prob_true],
            "prob_pred": [round(float(x), 4) for x in prob_pred],
        }
    except Exception:
        calib_data = {"prob_true": [], "prob_pred": []}

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "specificity": round(spec, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr],
            "tpr": [round(float(x), 4) for x in tpr],
        },
        "pr_curve": {
            "precision": [round(float(x), 4) for x in pr_prec],
            "recall": [round(float(x), 4) for x in pr_rec],
        },
        "calibration": calib_data,
        "sample_counts": {
            "total": int(len(y_true)),
            "positive_cases": n_pos,
            "negative_cases": n_neg,
            "prevalence": round(prevalence, 4),
        },
    }


def compare_models(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generates a clean comparison DataFrame from multiple model evaluation records."""
    rows = []
    for r in results_list:
        rows.append({
            "antibiotic": r["antibiotic"],
            "model": r["model"],
            "accuracy": r["metrics"]["accuracy"],
            "precision": r["metrics"]["precision"],
            "recall": r["metrics"]["recall"],
            "f1": r["metrics"]["f1"],
            "roc_auc": r["metrics"]["roc_auc"],
            "brier_score": r["metrics"]["brier_score"],
        })
    return pd.DataFrame(rows)


def run_evaluation_pipeline() -> Dict[str, Any]:
    """Executes the full evaluation pipeline across all 8 target antibiotics on the untouched test split."""
    logger.info("=" * 65)
    logger.info("AMR-Predict Standalone Model Evaluation Pipeline")
    logger.info("=" * 65)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset missing at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    X = df[MODEL_FEATURES].copy()

    prep_path = MODELS_DIR / "preprocessor.joblib"
    if not prep_path.exists():
        raise FileNotFoundError(f"Preprocessor missing at {prep_path}")
    preprocessor = load_preprocessor(str(prep_path))

    reg_path = ARTIFACTS_DIR / "model_registry.json"
    if not reg_path.exists():
        raise FileNotFoundError(f"Model registry missing at {reg_path}")
    with open(reg_path, "r") as f:
        registry = json.load(f)

    evaluation_report: Dict[str, Any] = {
        "metadata": {
            "evaluation_title": "AMR-X Model Evaluation on Untouched Holdout Test Partition",
            "dataset": "amr_synthetic_dataset.csv (20,000 episodes)",
            "split_methodology": {
                "test_partition": "20% stratified holdout (N=4,000)",
                "random_seed": 42,
                "leakage_isolation": "Untouched test data; strictly zero test labels used during preprocessor fitting or probability calibration.",
            },
            "limitations_statement": (
                "SYNTHETIC DATA RESEARCH PROTOTYPE: These evaluation metrics reflect performance on a "
                "synthetically generated cohort of 20,000 episodes. Real clinical antimicrobial susceptibility testing (AST) "
                "distributions and clinical outcomes will differ significantly. Models are NOT clinically validated."
            ),
            "medical_disclaimer": MEDICAL_DISCLAIMER,
        },
        "antibiotics": {},
    }

    detailed_metrics: Dict[str, Any] = {}
    comparison_records: List[Dict[str, Any]] = []

    for antibiotic, target_col in ANTIBIOTIC_TARGET_MAP.items():
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found. Skipping.")
            continue

        meta = registry.get(antibiotic, {})
        model_file = meta.get("model_file", f"{antibiotic.replace('-', '_')}_model.joblib")
        model_path = MODELS_DIR / model_file
        if not model_path.exists():
            raise FileNotFoundError(f"Model file missing: {model_path}")

        calibrated_model = joblib.load(model_path)
        selected_model_type = meta.get("model_type", "Random Forest")

        y = df[target_col].values.astype(int)

        # Re-derive exact untouched test split with fixed seed
        X_dev_raw, X_test_raw, y_dev, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        X_test = preprocessor.transform(X_test_raw)

        # Calibrated model test evaluation
        calib_prob = calibrated_model.predict_proba(X_test)[:, 1]
        metrics = evaluate_binary_classifier(y_test, calib_prob)

        # Extract base estimator if available for uncalibrated raw candidate comparison
        base_estimator = (
            calibrated_model.calibrated_classifiers_[0].estimator
            if hasattr(calibrated_model, "calibrated_classifiers_")
            else calibrated_model
        )
        raw_prob = base_estimator.predict_proba(X_test)[:, 1]
        raw_metrics = evaluate_binary_classifier(y_test, raw_prob)

        logger.info(
            f"{antibiotic:32} | Winner: {selected_model_type:14} | "
            f"ROC-AUC: {metrics['roc_auc']:.3f} | F1: {metrics['f1']:.3f} | "
            f"Recall: {metrics['recall']:.3f} | Precision: {metrics['precision']:.3f} | "
            f"Brier: {metrics['brier_score']:.3f}"
        )

        evaluation_report["antibiotics"][antibiotic] = {
            "selected_model": selected_model_type,
            "target_column": target_col,
            "model_file": model_file,
            "selection_score": meta.get("selection_score", 0.0),
            "validation_scores": meta.get("validation_scores", {}),
            "test_metrics": metrics,
            "uncalibrated_base_metrics": raw_metrics,
            "sample_counts": metrics["sample_counts"],
        }

        detailed_metrics[antibiotic] = {
            "selected_model": selected_model_type,
            "candidates": {
                selected_model_type: metrics,
            },
            "calibrated": metrics,
        }

        comparison_records.append({
            "antibiotic": antibiotic,
            "model": f"{selected_model_type} (Calibrated)",
            "metrics": metrics,
        })

    # Save artifacts/model_metrics.json
    out_metrics_path = ARTIFACTS_DIR / "model_metrics.json"
    with open(out_metrics_path, "w") as f:
        json.dump(evaluation_report, f, indent=2)
    logger.info(f"Saved comprehensive evaluation report to: {out_metrics_path}")

    # Synchronize artifacts/metrics.json for dashboard compatibility
    metrics_json_path = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(detailed_metrics, f, indent=2)
    logger.info(f"Synchronized dashboard metrics to: {metrics_json_path}")

    # Update comparison CSV if needed
    comp_df = compare_models(comparison_records)
    comp_df.to_csv(ARTIFACTS_DIR / "model_comparison.csv", index=False)
    logger.info("Updated artifacts/model_comparison.csv")

    logger.info("=" * 65)
    logger.info("Evaluation Pipeline Finished Successfully!")
    logger.info("=" * 65)

    return evaluation_report


if __name__ == "__main__":
    run_evaluation_pipeline()
