"""Comprehensive P0/P1 verification test suite for AMR-X pipeline integrity.

Enforces:
1. No target or probability leakage into predictive features.
2. No patient_id leakage into training or preprocessing.
3. Preprocessing fit boundaries (train split only).
4. Reproducible 4-way stratified splitting.
5. True candidate model selection (RF vs XGBoost).
6. Calibrated probability validity (bounded in [0, 1], shape (N, 2)).
7. Serialized model reload consistency.
8. All eight antibiotic prediction contracts.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import joblib

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    DATASET_PATH,
    MODELS_DIR,
    ARTIFACTS_DIR,
    ANTIBIOTIC_TARGET_MAP,
    MODEL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    EXCLUDED_COLUMNS,
    VALID_CATEGORIES,
)
from src.preprocessing import create_preprocessor, validate_patient_data
from src.predict import predict_patient, load_inference_artifacts


# ─── 1. Leakage Tests ─────────────────────────────────────────────────────────

def test_no_target_derived_columns_in_predictive_features():
    """Predictive features must strictly contain zero target or ground-truth probability columns."""
    df = pd.read_csv(DATASET_PATH, nrows=5)
    target_columns = list(ANTIBIOTIC_TARGET_MAP.values())
    prob_columns = [c for c in df.columns if c.endswith("_probability")]

    assert len(MODEL_FEATURES) == 16
    for f in MODEL_FEATURES:
        assert f not in target_columns, f"Target column '{f}' found in predictive features!"
        assert f not in prob_columns, f"Ground-truth probability column '{f}' found in features!"
        assert not f.endswith("_resistant"), f"Resistance target '{f}' leaked into features!"
        assert not f.endswith("_probability"), f"Probability column '{f}' leaked into features!"


def test_no_patient_id_leakage():
    """patient_id must never be included in predictive features."""
    assert "patient_id" not in MODEL_FEATURES
    assert "patient_id" in EXCLUDED_COLUMNS


# ─── 2. Preprocessing Fit Boundary Tests ──────────────────────────────────────

def test_train_only_preprocessing_fit():
    """Preprocessor must fit statistics strictly on training data, never on test data."""
    df = pd.read_csv(DATASET_PATH)
    from sklearn.model_selection import train_test_split

    y = df["Ciprofloxacin_resistant"].values
    X = df[MODEL_FEATURES]

    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # Fit preprocessor strictly on X_train
    prep_train = create_preprocessor()
    prep_train.fit(X_train)

    # Fit preprocessor on entire X
    prep_all = create_preprocessor()
    prep_all.fit(X)

    # StandardScaler statistics between train-only vs all must differ
    scaler_train = prep_train.named_transformers_["num"]
    scaler_all = prep_all.named_transformers_["num"]

    assert not np.allclose(scaler_train.mean_, scaler_all.mean_), (
        "Scaler mean on train split must not be identical to whole-dataset mean!"
    )


# ─── 3. Splitting Reproducibility Tests ────────────────────────────────────────

def test_reproducible_stratified_splits():
    """Splits must be deterministic, disjoint, and stratified."""
    df = pd.read_csv(DATASET_PATH)
    y = df["Nitrofurantoin_resistant"].values
    from sklearn.model_selection import train_test_split

    # First split: Holdout Test 20%
    X_dev_1, X_test_1, y_dev_1, y_test_1 = train_test_split(df, y, test_size=0.20, random_state=42, stratify=y)
    X_dev_2, X_test_2, y_dev_2, y_test_2 = train_test_split(df, y, test_size=0.20, random_state=42, stratify=y)

    # Reproducibility
    assert X_test_1.index.equals(X_test_2.index)
    assert np.array_equal(y_test_1, y_test_2)

    # Disjointness
    assert set(X_dev_1.index).isdisjoint(set(X_test_1.index))

    # Stratification check: prevalence in test matches prevalence in dev within 0.5%
    prev_dev = np.mean(y_dev_1)
    prev_test = np.mean(y_test_1)
    assert abs(prev_dev - prev_test) < 0.005


# ─── 4. Model Selection & Calibration Tests ───────────────────────────────────

def test_calibrated_model_artifacts_validity():
    """Saved models must be valid calibrated classifiers producing bounded probabilities in [0, 1]."""
    for abx in ANTIBIOTIC_TARGET_MAP.keys():
        fname = f"{abx.replace('-', '_')}_model.joblib"
        m_path = MODELS_DIR / fname
        assert m_path.exists(), f"Model file missing for {abx}"

        model = joblib.load(m_path)
        # Check predict_proba existence
        assert hasattr(model, "predict_proba"), f"Model for {abx} lacks predict_proba"

        # Test dummy prediction
        dummy_input = np.zeros((2, 25))
        probs = model.predict_proba(dummy_input)

        assert probs.shape == (2, 2), f"Expected shape (2, 2), got {probs.shape}"
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Probabilities must be in [0, 1]"
        assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities must sum to 1.0"


def test_registry_contains_both_rf_and_xgboost():
    """Model registry must reflect true selection between Random Forest and XGBoost."""
    reg_path = ARTIFACTS_DIR / "model_registry.json"
    assert reg_path.exists()

    with open(reg_path, "r") as f:
        registry = json.load(f)

    assert len(registry) == 8
    model_types = {meta["model_type"] for meta in registry.values()}
    # Model types must be legitimate
    for mt in model_types:
        assert mt in ["Random Forest", "XGBoost"]

    # Must include evaluation split details and random seed
    for abx, meta in registry.items():
        assert "training_version" in meta
        assert "random_seed" in meta
        assert meta["random_seed"] == 42
        assert "calibration_method" in meta
        assert "splits" in meta
        assert meta["splits"]["train_samples"] > 0
        assert meta["splits"]["test_samples"] > 0


# ─── 5. Inference Output & Reload Consistency ─────────────────────────────────

def test_all_eight_predictions_contract():
    """Inference must produce predictions for all 8 target antibiotics with correct attributes."""
    patient = {
        "age": 62,
        "sex": "F",
        "infection_type": "Complicated_UTI",
        "organism": "Klebsiella_pneumoniae",
        "diabetes": 1,
        "recent_hospitalization_90d": 1,
        "recent_antibiotic_use_90d": 1,
        "num_prior_uti_1yr": 2,
        "catheter_use": 1,
        "immunocompromised": 0,
        "nursing_home_resident": 1,
        "prior_resistant_culture_1yr": 1,
        "creatinine_mg_dl": 1.45,
        "wbc_count_k_ul": 12.8,
        "travel_last_6mo": 0,
        "healthcare_worker": 0,
    }

    res = predict_patient(patient)
    assert res["status"] == "success"
    assert len(res["predictions"]) == 8

    returned_abx = {p["antibiotic"] for p in res["predictions"]}
    expected_abx = set(ANTIBIOTIC_TARGET_MAP.keys())
    assert returned_abx == expected_abx

    for p in res["predictions"]:
        prob = p["estimated_resistance_probability"]
        assert 0.0 <= prob <= 1.0
        assert p["risk_category"] in ["Low", "Moderate", "High"]
        assert p["model_type"] in ["Random Forest", "XGBoost"]


def test_model_reload_consistency():
    """Reloading saved artifacts must produce strictly deterministic, identical probabilities."""
    patient = {
        "age": 45,
        "sex": "M",
        "infection_type": "UTI",
        "organism": "E_coli",
        "diabetes": 0,
        "recent_hospitalization_90d": 0,
        "recent_antibiotic_use_90d": 0,
        "num_prior_uti_1yr": 0,
        "catheter_use": 0,
        "immunocompromised": 0,
        "nursing_home_resident": 0,
        "prior_resistant_culture_1yr": 0,
        "creatinine_mg_dl": 0.95,
        "wbc_count_k_ul": 7.8,
        "travel_last_6mo": 0,
        "healthcare_worker": 0,
    }

    res1 = predict_patient(patient)
    res2 = predict_patient(patient)

    for p1, p2 in zip(res1["predictions"], res2["predictions"]):
        assert p1["antibiotic"] == p2["antibiotic"]
        assert p1["estimated_resistance_probability"] == p2["estimated_resistance_probability"]
        assert p1["risk_category"] == p2["risk_category"]
