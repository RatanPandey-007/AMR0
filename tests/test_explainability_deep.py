"""In-depth verification test suite for Phase 2 Explainability (SHAP TreeExplainer).

Enforces:
1. True TreeExplainer execution for all 8 target models.
2. Numerical additivity axiom: sum(SHAP) + base_val == explained_model_output (within 1e-4).
3. Feature ordering consistency (positive, negative, absolute magnitude).
4. Deterministic explanation stability.
5. Target selection differentiation.
6. Clear separation between native tree output and calibrated risk probabilities.
7. Anti-causality attribution disclaimers.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import ANTIBIOTIC_TARGET_MAP
from src.explain import explain_patient_prediction, get_global_shap_summary


@pytest.fixture
def sample_patient():
    return {
        "age": 68,
        "sex": "F",
        "infection_type": "Catheter_Associated_UTI",
        "organism": "Klebsiella_pneumoniae",
        "diabetes": 1,
        "recent_hospitalization_90d": 1,
        "recent_antibiotic_use_90d": 1,
        "num_prior_uti_1yr": 3,
        "catheter_use": 1,
        "immunocompromised": 0,
        "nursing_home_resident": 1,
        "prior_resistant_culture_1yr": 1,
        "creatinine_mg_dl": 1.75,
        "wbc_count_k_ul": 14.2,
        "travel_last_6mo": 0,
        "healthcare_worker": 0,
    }


def test_all_eight_models_shap_explainability(sample_patient):
    """Every antibiotic model must produce mathematically sound SHAP explanations."""
    for abx in ANTIBIOTIC_TARGET_MAP.keys():
        exp = explain_patient_prediction(sample_patient, abx)

        assert exp["antibiotic"] == abx
        assert exp["model_type"] in ["Random Forest", "XGBoost"]
        assert exp["method"] == "SHAP TreeExplainer"
        assert exp["output_space"] in ["probability", "margin (log-odds)"]

        # Additivity verification: sum(SHAP) + base_value == explained_model_output
        additivity = exp["additivity_check"]
        assert additivity["verified"] is True
        assert additivity["difference"] < 1e-4, f"Additivity violated for {abx}: diff={additivity['difference']}"

        # Calibrated probability must be bounded [0, 1]
        calib_p = exp["estimated_resistance_probability"]
        assert 0.0 <= calib_p <= 1.0


def test_feature_ordering_and_top_factors(sample_patient):
    """Waterfall features must be sorted descending by absolute magnitude; pos/neg factors properly filtered."""
    exp = explain_patient_prediction(sample_patient, "Ciprofloxacin")

    # Waterfall features sorted by abs_shap descending
    waterfall = exp["waterfall_features"]
    assert len(waterfall) > 0
    abs_shaps = [f["abs_shap"] for f in waterfall]
    assert abs_shaps == sorted(abs_shaps, reverse=True)

    # Positive factors strictly positive and descending
    pos_factors = exp["top_positive_factors"]
    for f in pos_factors:
        assert f["shap_value"] > 0
    pos_shaps = [f["shap_value"] for f in pos_factors]
    assert pos_shaps == sorted(pos_shaps, reverse=True)

    # Negative factors strictly negative and ascending (most negative first)
    neg_factors = exp["top_negative_factors"]
    for f in neg_factors:
        assert f["shap_value"] < 0
    neg_shaps = [f["shap_value"] for f in neg_factors]
    assert neg_shaps == sorted(neg_shaps)


def test_explanation_stability_and_determinism(sample_patient):
    """Successive calls for the exact same patient must produce identical SHAP values."""
    exp1 = explain_patient_prediction(sample_patient, "Nitrofurantoin")
    exp2 = explain_patient_prediction(sample_patient, "Nitrofurantoin")

    assert exp1["base_value"] == exp2["base_value"]
    assert exp1["explained_model_output"] == exp2["explained_model_output"]
    assert exp1["estimated_resistance_probability"] == exp2["estimated_resistance_probability"]

    shaps1 = [f["shap_value"] for f in exp1["waterfall_features"]]
    shaps2 = [f["shap_value"] for f in exp2["waterfall_features"]]
    assert np.allclose(shaps1, shaps2, atol=1e-6)


def test_target_selection_differentiation(sample_patient):
    """Explanations for different antibiotics must differ reflecting distinct model weights."""
    exp_cipro = explain_patient_prediction(sample_patient, "Ciprofloxacin")
    exp_fosfo = explain_patient_prediction(sample_patient, "Fosfomycin")

    assert exp_cipro["antibiotic"] != exp_fosfo["antibiotic"]
    assert exp_cipro["base_value"] != exp_fosfo["base_value"] or exp_cipro["explained_model_output"] != exp_fosfo["explained_model_output"]


def test_global_shap_summary_all_eight_targets():
    """Global feature summary must cover all 8 targets with valid mean SHAP importances."""
    summary = get_global_shap_summary()
    assert len(summary) == 8

    for abx in ANTIBIOTIC_TARGET_MAP.keys():
        assert abx in summary
        meta = summary[abx]
        assert "top_features" in meta
        assert len(meta["top_features"]) >= 10
        # Verify descending order of mean_shap
        vals = [f["mean_shap"] for f in meta["top_features"]]
        assert vals == sorted(vals, reverse=True)
        assert all(v >= 0.0 for v in vals)


def test_correlation_vs_causation_disclaimer_present(sample_patient):
    """Output must include an explicit disclaimer warning that SHAP reflects statistical association, not biological causality."""
    exp = explain_patient_prediction(sample_patient, "Levofloxacin")
    notice = exp["attribution_notice"].lower()
    assert "not" in notice
    assert "causality" in notice
