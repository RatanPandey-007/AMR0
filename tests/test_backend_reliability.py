"""Automated tests for Phase 4: Backend Reliability, API Validation, Database Persistence, and PDF Generation."""

import os
import tempfile
import pytest
import sqlite3
from pathlib import Path
from fastapi.testclient import TestClient

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.main import app
from src.database import init_db, get_db_path, save_prediction, get_prediction_by_id, delete_prediction
from src.report_generator import generate_pdf_report

client = TestClient(app)

SAMPLE_VALID_PATIENT = {
    "patient_id": "PT-TEST-001",
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
    "nursing_home_resident": 0,
    "prior_resistant_culture_1yr": 1,
    "creatinine_mg_dl": 1.4,
    "wbc_count_k_ul": 12.8,
    "travel_last_6mo": 0,
    "healthcare_worker": 0,
}


def test_api_input_validation_invalid_age():
    """Verify API rejects out-of-bound age with HTTP 422."""
    bad_payload = dict(SAMPLE_VALID_PATIENT)
    bad_payload["age"] = 150  # Over 120
    res = client.post("/api/predict", json=bad_payload)
    assert res.status_code == 422


def test_api_input_validation_negative_labs():
    """Verify API rejects invalid or negative laboratory values."""
    bad_creatinine = dict(SAMPLE_VALID_PATIENT)
    bad_creatinine["creatinine_mg_dl"] = -1.5
    res = client.post("/api/predict", json=bad_creatinine)
    assert res.status_code == 422

    bad_wbc = dict(SAMPLE_VALID_PATIENT)
    bad_wbc["wbc_count_k_ul"] = 0.0  # Below min 0.1
    res = client.post("/api/predict", json=bad_wbc)
    assert res.status_code == 422


def test_api_input_validation_invalid_categories():
    """Verify API rejects unwhitelisted categorical values."""
    bad_sex = dict(SAMPLE_VALID_PATIENT)
    bad_sex["sex"] = "UNKNOWN"
    res = client.post("/api/predict", json=bad_sex)
    assert res.status_code == 422

    bad_organism = dict(SAMPLE_VALID_PATIENT)
    bad_organism["organism"] = "Superbug_fake"
    res = client.post("/api/predict", json=bad_organism)
    assert res.status_code == 422

    bad_syndrome = dict(SAMPLE_VALID_PATIENT)
    bad_syndrome["infection_type"] = "Respiratory_Infection"
    res = client.post("/api/predict", json=bad_syndrome)
    assert res.status_code == 422


def test_prediction_api_and_persistence_roundtrip():
    """Verify prediction produces all 8 antibiotic results and saves to database."""
    res = client.post("/api/predict", json=SAMPLE_VALID_PATIENT)
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "success"
    assert "prediction_id" in data
    pred_id = data["prediction_id"]
    assert pred_id.startswith("PRD-")
    assert len(data["predictions"]) == 8

    # Fetch from history endpoint
    hist_res = client.get(f"/api/history/{pred_id}")
    assert hist_res.status_code == 200
    hist_data = hist_res.json()

    assert hist_data["id"] == pred_id
    assert hist_data["patient_id"] == "PT-TEST-001"
    assert hist_data["organism"] == "Klebsiella_pneumoniae"
    assert len(hist_data["predictions"]) == 8

    # Verify report JSON endpoint
    rpt_res = client.get(f"/api/reports/{pred_id}")
    assert rpt_res.status_code == 200
    rpt_data = rpt_res.json()
    assert rpt_data["prediction_id"] == pred_id
    assert "disclaimer" in rpt_data
    assert "patient_profile" in rpt_data

    # Verify report PDF download endpoint
    pdf_res = client.get(f"/api/reports/{pred_id}/pdf")
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert pdf_res.content.startswith(b"%PDF-")
    assert len(pdf_res.content) > 1000

    # Cleanup: delete record
    del_res = client.delete(f"/api/history/{pred_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"


def test_pdf_report_generator_direct():
    """Verify direct PDF generation produces compliant bytes and structure."""
    mock_record = {
        "id": "PRD-MOCK-99",
        "patient_id": "PT-MOCK-99",
        "created_at": "2026-11-01 12:00:00",
        "age": 70,
        "sex": "M",
        "infection_type": "Pyelonephritis",
        "organism": "E_coli",
        "creatinine_mg_dl": 1.5,
        "wbc_count_k_ul": 14.2,
        "highest_risk_antibiotic": "Ciprofloxacin",
        "highest_risk_prob": 0.65,
        "highest_risk_category": "High",
        "clinical_factors": ["Diabetes Mellitus", "Hospitalization (Last 90d)"],
        "predictions": [
            {
                "antibiotic": "Ciprofloxacin",
                "model_type": "Random Forest",
                "estimated_resistance_probability": 0.65,
                "risk_category": "High",
                "interpretation_label": "Elevated resistance likelihood",
            },
            {
                "antibiotic": "Nitrofurantoin",
                "model_type": "XGBoost",
                "estimated_resistance_probability": 0.15,
                "risk_category": "Low",
                "interpretation_label": "High susceptibility likelihood",
            },
        ],
        "model_version": "2.0.0",
        "execution_time_ms": 12.5,
    }

    pdf_bytes = generate_pdf_report(mock_record)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 2000


def test_custom_db_path_environment_override():
    """Verify AMR_DB_PATH environment variable correctly overrides the database path."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        test_db_path = Path(tmp_dir) / "custom_test_amr.db"
        orig_env = os.environ.get("AMR_DB_PATH")
        try:
            os.environ["AMR_DB_PATH"] = str(test_db_path)
            assert get_db_path() == test_db_path
            init_db()
            assert test_db_path.exists()

            # Verify schema was initialized
            conn = sqlite3.connect(test_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions';")
            table = cursor.fetchone()
            assert table is not None
            conn.close()
        finally:
            if orig_env is not None:
                os.environ["AMR_DB_PATH"] = orig_env
            else:
                os.environ.pop("AMR_DB_PATH", None)
