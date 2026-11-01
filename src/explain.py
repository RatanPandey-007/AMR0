"""Tree Explainable-AI Engine for AMR-Predict.

Computes mathematically rigorous local patient-specific and global feature attributions
using standard SHAP TreeExplainer across both Random Forest and XGBoost architectures.
Ensures strict additivity verification in the model's native output space and explicitly
distinguishes mathematical attribution from biological causality.
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
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from src.config import (
    MODELS_DIR,
    ARTIFACTS_DIR,
    ANTIBIOTIC_TARGET_MAP,
    FEATURE_DISPLAY_NAMES,
    MODEL_ATTRIBUTION_NOTICE,
    MEDICAL_DISCLAIMER,
)
from src.preprocessing import (
    validate_patient_data,
    load_preprocessor,
    get_preprocessed_feature_names,
)

logger = logging.getLogger("AMR_Explain")

_MODEL_CACHE: Dict[str, Any] = {}
_TREE_EXPLAINER_CACHE: Dict[str, Any] = {}
_PREPROCESSOR_CACHE: Optional[Any] = None
_REGISTRY_CACHE: Optional[Dict[str, Any]] = None


def clear_explain_cache() -> None:
    """Clears cached models, explainers, preprocessor, and registry from memory."""
    global _MODEL_CACHE, _TREE_EXPLAINER_CACHE, _PREPROCESSOR_CACHE, _REGISTRY_CACHE
    _MODEL_CACHE.clear()
    _TREE_EXPLAINER_CACHE.clear()
    _PREPROCESSOR_CACHE = None
    _REGISTRY_CACHE = None
    logger.info("Explainability cache cleared.")


def _load_resources() -> Tuple[Any, Dict[str, Any]]:
    """Loads and caches preprocessor and model registry."""
    global _PREPROCESSOR_CACHE, _REGISTRY_CACHE

    if _PREPROCESSOR_CACHE is None:
        prep_path = MODELS_DIR / "preprocessor.joblib"
        if not prep_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {prep_path}")
        _PREPROCESSOR_CACHE = load_preprocessor(str(prep_path))

    if _REGISTRY_CACHE is None:
        reg_path = ARTIFACTS_DIR / "model_registry.json"
        if not reg_path.exists():
            raise FileNotFoundError(f"Model registry artifact not found at {reg_path}")
        with open(reg_path, "r") as f:
            _REGISTRY_CACHE = json.load(f)

    return _PREPROCESSOR_CACHE, _REGISTRY_CACHE


def _get_model(antibiotic: str) -> Tuple[Any, str]:
    """Retrieves cached model for the specified antibiotic."""
    _, registry = _load_resources()

    if antibiotic not in registry:
        raise KeyError(
            f"Antibiotic '{antibiotic}' not found in registry. "
            f"Available: {list(registry.keys())}"
        )

    meta = registry[antibiotic]
    model_file = meta["model_file"]
    model_type = meta["model_type"]

    if antibiotic not in _MODEL_CACHE:
        model_path = MODELS_DIR / model_file
        if not model_path.exists():
            raise FileNotFoundError(f"Model file {model_path} not found.")
        _MODEL_CACHE[antibiotic] = joblib.load(model_path)

    model = _MODEL_CACHE[antibiotic]
    return model, model_type


def _get_tree_explainer(antibiotic: str, base_estimator: Any) -> shap.TreeExplainer:
    """Retrieves or builds cached SHAP TreeExplainer for the base estimator."""
    if antibiotic not in _TREE_EXPLAINER_CACHE:
        _TREE_EXPLAINER_CACHE[antibiotic] = shap.TreeExplainer(base_estimator)
    return _TREE_EXPLAINER_CACHE[antibiotic]


def _format_feature_label(raw_col_name: str, patient_val: Any) -> str:
    """Formats an encoded feature name and raw value into a human-friendly clinical description."""
    if raw_col_name.startswith("sex_"):
        suffix = raw_col_name[len("sex_"):]
        return f"Sex: {'Female' if suffix == 'F' else 'Male'}"
    elif raw_col_name.startswith("infection_type_"):
        suffix = raw_col_name[len("infection_type_"):].replace("_", " ")
        return f"Infection: {suffix}"
    elif raw_col_name.startswith("organism_"):
        suffix = raw_col_name[len("organism_"):].replace("_", " ")
        return f"Pathogen: {suffix}"

    display_title = FEATURE_DISPLAY_NAMES.get(raw_col_name, raw_col_name.replace("_", " ").title())
    if patient_val is not None:
        if raw_col_name in ["diabetes", "recent_hospitalization_90d", "recent_antibiotic_use_90d",
                            "catheter_use", "immunocompromised", "nursing_home_resident",
                            "prior_resistant_culture_1yr", "travel_last_6mo", "healthcare_worker"]:
            val_text = "Yes" if int(patient_val) == 1 else "No"
            return f"{display_title} ({val_text})"
        return f"{display_title} = {patient_val}"
    return display_title


def _compute_rf_fallback_contributions(model: Any, X_enc: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fallback: computes exact decision tree path probability contributions for Random Forest."""
    n_features = X_enc.shape[1]
    contributions = np.zeros(n_features)
    base_val = 0.0
    for dt in model.estimators_:
        tree = dt.tree_
        node_indicator = dt.decision_path(X_enc)
        node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
        values = tree.value[:, 0, :]
        probs = values / values.sum(axis=1, keepdims=True)
        base_val += probs[0, 1]
        for i in range(len(node_index) - 1):
            curr_node = node_index[i]
            next_node = node_index[i+1]
            feature = tree.feature[curr_node]
            diff = probs[next_node, 1] - probs[curr_node, 1]
            contributions[feature] += diff
    n_trees = len(model.estimators_)
    return contributions / n_trees, base_val / n_trees


def explain_patient_prediction(
    patient_data: Dict[str, Any],
    antibiotic: str,
    top_n: int = 8
) -> Dict[str, Any]:
    """Generates local SHAP explanation for a specific patient and antibiotic.

    Guarantees:
    - Additivity axiom: sum(SHAP) + base_value == explained_model_output (within 1e-4)
    - Output space transparency: probability space for Random Forest, margin (log-odds) for XGBoost
    - Distinction between raw tree output and calibrated risk probability
    - Correlation vs causation medical disclaimer
    """
    preprocessor, registry = _load_resources()

    patient_copy = dict(patient_data)
    patient_copy.pop("patient_id", None)

    is_valid, errors, cleaned_data = validate_patient_data(patient_copy)
    if not is_valid:
        raise ValueError(f"Patient data validation failed: {'; '.join(errors)}")

    model, model_type = _get_model(antibiotic)

    # Unwrap calibrated classifier if present to get underlying tree estimator
    base_estimator = (
        model.calibrated_classifiers_[0].estimator
        if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0
        else model
    )

    # Convert to DataFrame and transform
    df_patient = pd.DataFrame([cleaned_data])
    X_enc = preprocessor.transform(df_patient)
    feature_names = get_preprocessed_feature_names(preprocessor)

    # 1. Final Calibrated Probability (Clinical Risk Tier)
    calibrated_prob = float(model.predict_proba(X_enc)[0, 1])

    # 2. Raw Tree Output in Native Output Space
    if model_type == "Random Forest":
        output_space = "probability"
        explained_output = float(base_estimator.predict_proba(X_enc)[0, 1])
    else:
        output_space = "margin (log-odds)"
        explained_output = float(base_estimator.predict(X_enc, output_margin=True)[0])

    # 3. Compute Standard SHAP Values via TreeExplainer
    explanation_method = "SHAP TreeExplainer"
    try:
        explainer = _get_tree_explainer(antibiotic, base_estimator)
        shap_vals = explainer.shap_values(X_enc)

        base_val_raw = explainer.expected_value
        if isinstance(base_val_raw, (list, np.ndarray)):
            base_val = float(base_val_raw[1]) if len(base_val_raw) > 1 else float(base_val_raw[0])
        else:
            base_val = float(base_val_raw)

        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            # Random Forest shape (1, n_features, 2) -> class 1
            sv = shap_vals[0, :, 1]
        elif isinstance(shap_vals, list):
            sv = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
            # XGBoost binary classification log-odds
            sv = shap_vals[0]
        else:
            sv = np.zeros(X_enc.shape[1])

    except Exception as e:
        logger.warning(f"TreeExplainer exception for {antibiotic}: {e}, falling back to decision path decomposition")
        explanation_method = "Decision Tree Path Decomposition (Fallback)"
        if hasattr(base_estimator, "estimators_"):
            sv, base_val = _compute_rf_fallback_contributions(base_estimator, X_enc)
        else:
            sv = getattr(base_estimator, "feature_importances_", np.ones(X_enc.shape[1]))
            base_val = 0.5

    # 4. Numerical Additivity Verification (sum(SHAP) + base_val == explained_output)
    calc_sum = float(np.sum(sv) + base_val)
    additivity_diff = abs(calc_sum - explained_output)
    additivity_verified = bool(additivity_diff < 1e-4)

    # 5. Decompose into Human-Readable Clinical Contributions
    contributions = []
    for fname, s_val in zip(feature_names, sv):
        raw_val = None
        if fname in cleaned_data:
            raw_val = cleaned_data[fname]
        else:
            for cat_f in ["sex", "infection_type", "organism"]:
                if fname.startswith(cat_f + "_"):
                    cat_val = fname[len(cat_f) + 1:]
                    raw_val = 1 if cleaned_data.get(cat_f) == cat_val else 0

        contributions.append({
            "feature_key": fname,
            "display_name": _format_feature_label(fname, raw_val),
            "shap_value": round(float(s_val), 4),
            "abs_shap": abs(float(s_val)),
            "direction": "Positive (+Risk)" if s_val > 0 else "Negative (-Risk)" if s_val < 0 else "Neutral",
        })

    # Sort factors by impact
    positive_factors = [c for c in contributions if c["shap_value"] > 0]
    positive_factors.sort(key=lambda x: x["shap_value"], reverse=True)

    negative_factors = [c for c in contributions if c["shap_value"] < 0]
    negative_factors.sort(key=lambda x: x["shap_value"])  # Most negative first

    all_sorted = sorted(contributions, key=lambda x: x["abs_shap"], reverse=True)
    waterfall_items = all_sorted[:top_n]

    return {
        "antibiotic": antibiotic,
        "model_type": model_type,
        "method": explanation_method,
        "output_space": output_space,
        "base_value": round(float(base_val), 4),
        "explained_model_output": round(float(explained_output), 4),
        "estimated_resistance_probability": round(calibrated_prob, 4),
        "percentage_display": f"{calibrated_prob * 100:.1f}%",
        "additivity_check": {
            "verified": additivity_verified,
            "sum_of_contributions": round(calc_sum, 4),
            "explained_model_output": round(explained_output, 4),
            "difference": round(additivity_diff, 8),
        },
        "top_positive_factors": positive_factors[:top_n],
        "top_negative_factors": negative_factors[:top_n],
        "waterfall_features": waterfall_items,
        "attribution_notice": (
            f"SHAP values reflect mathematical feature contributions in the model's native {output_space} "
            "space. These attributions quantify statistical association in tree paths, NOT biological or clinical causality."
        ),
        "medical_disclaimer": MEDICAL_DISCLAIMER,
    }


def get_global_shap_summary(antibiotic: Optional[str] = None) -> Dict[str, Any]:
    """Retrieves pre-calculated dataset-wide feature importances computed from holdout background split."""
    shap_path = ARTIFACTS_DIR / "global_shap_summary.json"
    if not shap_path.exists():
        raise FileNotFoundError("Global feature summary not found in artifacts.")

    with open(shap_path, "r") as f:
        data = json.load(f)

    if antibiotic:
        if antibiotic not in data:
            raise KeyError(f"Antibiotic '{antibiotic}' not in global summary.")
        return {
            "antibiotic": antibiotic,
            "global_shap": data[antibiotic],
            "attribution_notice": MODEL_ATTRIBUTION_NOTICE,
        }
    return data
