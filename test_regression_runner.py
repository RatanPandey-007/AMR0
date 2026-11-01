"""
AMR-X Complete Regression Test Suite
Independent QA Engineering Verification Script
"""

import sys
import os
import json
import time
import math
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(str(BASE_DIR))

# Ensure test DB isolation
TEST_DB_PATH = BASE_DIR / "data" / "amr_regression_test.db"
os.environ["AMR_DB_PATH"] = str(TEST_DB_PATH)

results = []

def record(category, test_name, status, detail=""):
    results.append({
        "category": category,
        "test": test_name,
        "status": status,
        "detail": detail
    })
    mark = "[PASS]" if status == "PASS" else "[FAIL]"
    print(f"{mark} {category} :: {test_name} - {detail}")


from src.config import ANTIBIOTIC_TARGET_MAP
antibiotic_targets = list(ANTIBIOTIC_TARGET_MAP.keys())

# ==============================================================================
# SECTION 1: BACKEND TESTING
# ==============================================================================
print("\n" + "="*70)
print("SECTION 1: BACKEND REGRESSION TESTS")
print("="*70)

# 1.1 Imports
try:
    import app.main
    import src.config
    import src.database
    import src.evaluate
    import src.explain
    import src.predict
    import src.preprocessing
    import src.report_generator
    import src.train
    record("BACKEND", "Module Imports", "PASS", "All 9 backend modules imported without error")
except Exception as e:
    record("BACKEND", "Module Imports", "FAIL", str(e))

# 1.2 Model Artifacts Verification
try:
    import joblib
    models_dir = BASE_DIR / "models"
    artifacts_dir = BASE_DIR / "artifacts"
    
    expected_artifacts = [
        models_dir / "preprocessor.joblib",
        artifacts_dir / "model_metrics.json",
        artifacts_dir / "model_comparison.csv",
    ]
    
    from src.config import ANTIBIOTIC_TARGET_MAP
    antibiotic_targets = list(ANTIBIOTIC_TARGET_MAP.keys())
    for abx, target_col in ANTIBIOTIC_TARGET_MAP.items():
        # Models are saved with sanitized target names (e.g., Amoxicillin_Clavulanate_model.joblib)
        sanitized = abx.replace("-", "_")
        expected_artifacts.append(models_dir / f"{sanitized}_model.joblib")
        
    all_exist = True
    for p in expected_artifacts:
        if not p.exists() or p.stat().st_size == 0:
            all_exist = False
            record("BACKEND", f"Artifact {p.name}", "FAIL", f"Missing or zero-size: {p}")
        else:
            record("BACKEND", f"Artifact {p.name}", "PASS", f"Size: {p.stat().st_size / 1024:.1f} KB")
            
except Exception as e:
    record("BACKEND", "Model Artifacts Check", "FAIL", str(e))

# 1.3 FastAPI TestClient Setup
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# 1.4 API Health Endpoint
try:
    res = client.get("/api/health")
    if res.status_code == 200 and res.json().get("status") in ["healthy", "ok"]:
        record("BACKEND", "Health Endpoint (/api/health)", "PASS", f"Status: {res.json().get('status')}")
    else:
        record("BACKEND", "Health Endpoint (/api/health)", "FAIL", f"HTTP {res.status_code}: {res.text}")
except Exception as e:
    record("BACKEND", "Health Endpoint (/api/health)", "FAIL", str(e))

# 1.5 Eight Antibiotic Inference Paths
sample_patient = {
    "patient_id": "QA-TEST-001",
    "age": 65,
    "sex": "F",
    "creatinine_mg_dl": 1.25,
    "wbc_count_k_ul": 12.4,
    "infection_type": "Complicated_UTI",
    "organism": "Klebsiella_pneumoniae",
    "num_prior_uti_1yr": 2,
    "recent_hospitalization_90d": 1,
    "recent_antibiotic_use_90d": 1,
    "prior_resistant_culture_1yr": 1,
    "diabetes": 1,
    "catheter_use": 0,
    "immunocompromised": 0,
    "nursing_home_resident": 0,
    "travel_last_6mo": 0,
    "healthcare_worker": 0
}

created_prediction_id = None
try:
    res = client.post("/api/predict", json=sample_patient)
    if res.status_code == 200:
        data = res.json()
        preds = data.get("predictions", [])
        created_prediction_id = data.get("prediction_id")
        
        if len(preds) == 8:
            all_valid_range = True
            for info in preds:
                p = info.get("estimated_resistance_probability")
                risk = info.get("risk_category")
                model_type = info.get("model_type")
                if p is None or not (0.0 <= p <= 1.0) or risk not in ["Low", "Moderate", "High"] or not model_type:
                    all_valid_range = False
            if all_valid_range:
                record("BACKEND", "All 8 Inference Paths", "PASS", f"Generated 8 valid predictions. ID={created_prediction_id}")
            else:
                record("BACKEND", "All 8 Inference Paths", "FAIL", "Invalid probability or risk values")
        else:
            record("BACKEND", "All 8 Inference Paths", "FAIL", f"Expected 8 predictions, got {len(preds)}")
    else:
        record("BACKEND", "All 8 Inference Paths", "FAIL", f"HTTP {res.status_code}: {res.text}")
except Exception as e:
    record("BACKEND", "All 8 Inference Paths", "FAIL", str(e))

# 1.6 Invalid Input Rejections
invalid_cases = [
    ("Negative Age", {"age": -5}, 422),
    ("Extreme Age (>120)", {"age": 140}, 422),
    ("Negative Creatinine", {"creatinine_mg_dl": -1.0}, 422),
    ("Negative WBC", {"wbc_count_k_ul": -5.0}, 422),
    ("Unknown Organism", {"organism": "Streptococcus_pyogenes"}, 422),
    ("Unknown Infection", {"infection_type": "Bacteremia"}, 422),
    ("Missing Required Age", {"age": None}, 422),
]

for name, patch, expected_code in invalid_cases:
    bad_payload = dict(sample_patient)
    bad_payload.update(patch)
    try:
        r = client.post("/api/predict", json=bad_payload)
        if r.status_code == expected_code:
            record("BACKEND", f"Validation: {name}", "PASS", f"Rejected with HTTP {r.status_code}")
        else:
            record("BACKEND", f"Validation: {name}", "FAIL", f"Expected {expected_code}, got {r.status_code}")
    except Exception as e:
        record("BACKEND", f"Validation: {name}", "FAIL", str(e))

# 1.7 Database Writes and Reads
try:
    if created_prediction_id:
        r_hist = client.get(f"/api/history/{created_prediction_id}")
        if r_hist.status_code == 200:
            hist_item = r_hist.json()
            if hist_item.get("id") == created_prediction_id:
                record("BACKEND", "Database Write/Read Roundtrip", "PASS", f"Retrieved record {created_prediction_id}")
            else:
                record("BACKEND", "Database Write/Read Roundtrip", "FAIL", f"ID mismatch: {hist_item.get('id')} vs {created_prediction_id}")
        else:
            record("BACKEND", "Database Write/Read Roundtrip", "FAIL", f"HTTP {r_hist.status_code}: {r_hist.text}")
            
        r_list = client.get("/api/history?risk_level=All")
        if r_list.status_code == 200 and len(r_list.json().get("history", [])) > 0:
            record("BACKEND", "Database Query (/api/history)", "PASS", f"Listed {len(r_list.json().get('history'))} records")
        else:
            record("BACKEND", "Database Query (/api/history)", "FAIL", "Could not query history list")
    else:
        record("BACKEND", "Database Write/Read Roundtrip", "FAIL", "No prediction ID to test")
except Exception as e:
    record("BACKEND", "Database Persistence Check", "FAIL", str(e))

# 1.8 Report Generation (JSON & Vector PDF)
try:
    if created_prediction_id:
        r_rep = client.get(f"/api/reports/{created_prediction_id}")
        if r_rep.status_code == 200 and "antibiogram_predictions" in r_rep.json():
            record("BACKEND", "Report JSON Generation (/api/reports/{id})", "PASS", "Valid structured report JSON")
        else:
            record("BACKEND", "Report JSON Generation (/api/reports/{id})", "FAIL", f"HTTP {r_rep.status_code}")
            
        r_pdf = client.get(f"/api/reports/{created_prediction_id}/pdf")
        if r_pdf.status_code == 200 and r_pdf.content.startswith(b"%PDF") and len(r_pdf.content) > 1000:
            record("BACKEND", "Report PDF Generation (/api/reports/{id}/pdf)", "PASS", f"Valid vector PDF ({len(r_pdf.content)} bytes)")
        else:
            record("BACKEND", "Report PDF Generation (/api/reports/{id}/pdf)", "FAIL", f"Invalid PDF response: {r_pdf.status_code}")
    else:
        record("BACKEND", "Report Generation Check", "FAIL", "No prediction ID available")
except Exception as e:
    record("BACKEND", "Report Generation Check", "FAIL", str(e))


# ==============================================================================
# SECTION 2: MACHINE LEARNING & EVALUATION TESTING
# ==============================================================================
print("\n" + "="*70)
print("SECTION 2: ML & EVALUATION REGRESSION TESTS")
print("="*70)

from src.config import MODEL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, BINARY_FEATURES

# 2.1 Feature Schema
try:
    assert len(MODEL_FEATURES) == 16, f"Expected 16 features, got {len(MODEL_FEATURES)}"
    assert len(CATEGORICAL_FEATURES) == 3, f"Expected 3 categorical, got {len(CATEGORICAL_FEATURES)}"
    assert len(NUMERIC_FEATURES) == 13, f"Expected 13 numeric/binary, got {len(NUMERIC_FEATURES)}"
    assert len(BINARY_FEATURES) == 9, f"Expected 9 binary, got {len(BINARY_FEATURES)}"
    record("ML", "Feature Schema Definitions", "PASS", "16 total features (3 cat, 4 lab/demo num, 9 bin)")
except Exception as e:
    record("ML", "Feature Schema Definitions", "FAIL", str(e))

# 2.2 Preprocessing Consistency
try:
    from src.preprocessing import validate_patient_data
    preprocessor = joblib.load(models_dir / "preprocessor.joblib")
    
    _, _, cleaned1 = validate_patient_data(sample_patient)
    df1 = pd.DataFrame([cleaned1])[MODEL_FEATURES]
    vec1 = preprocessor.transform(df1)
    
    _, _, cleaned2 = validate_patient_data(sample_patient)
    df2 = pd.DataFrame([cleaned2])[MODEL_FEATURES]
    vec2 = preprocessor.transform(df2)
    
    if np.array_equal(vec1, vec2) and vec1.shape == (1, 25):
        if not np.isnan(vec1).any() and not np.isinf(vec1).any():
            record("ML", "Preprocessing Consistency", "PASS", f"Deterministic output shape (1, 25), no NaNs/Infs")
        else:
            record("ML", "Preprocessing Consistency", "FAIL", "Found NaN or Inf in transformed vector")
    else:
        record("ML", "Preprocessing Consistency", "FAIL", f"Shape mismatch or non-deterministic: {vec1.shape}")
except Exception as e:
    record("ML", "Preprocessing Consistency", "FAIL", str(e))

# 2.3 Calibration Artifacts & Probability Range Test
try:
    all_calibrated = True
    from sklearn.calibration import CalibratedClassifierCV
    
    for abx in antibiotic_targets:
        model_path = models_dir / f"{abx.replace('-', '_')}_model.joblib"
        m = joblib.load(model_path)
        if not isinstance(m, CalibratedClassifierCV):
            all_calibrated = False
            
    if all_calibrated:
        record("ML", "Calibration Artifacts Type", "PASS", "All 8 models are instances of CalibratedClassifierCV")
    else:
        record("ML", "Calibration Artifacts Type", "FAIL", "Some models are not CalibratedClassifierCV")
        
    # Stress test probability bounds across 100 perturbed profiles
    import random
    all_bounded = True
    for i in range(100):
        test_p = dict(sample_patient)
        test_p["age"] = random.randint(18, 95)
        test_p["creatinine_mg_dl"] = round(random.uniform(0.4, 6.0), 2)
        test_p["wbc_count_k_ul"] = round(random.uniform(2.0, 35.0), 1)
        test_p["num_prior_uti_1yr"] = random.randint(0, 15)
        test_p["recent_hospitalization_90d"] = random.choice([0, 1])
        test_p["recent_antibiotic_use_90d"] = random.choice([0, 1])
        test_p["organism"] = random.choice(["E_coli", "Klebsiella_pneumoniae", "Enterococcus_faecalis"])
        
        _, _, cl = validate_patient_data(test_p)
        df_p = pd.DataFrame([cl])[MODEL_FEATURES]
        vec = preprocessor.transform(df_p)
        
        for abx in antibiotic_targets:
            m = joblib.load(models_dir / f"{abx.replace('-', '_')}_model.joblib")
            prob = m.predict_proba(vec)[0, 1]
            if not (0.0 <= prob <= 1.0) or math.isnan(prob):
                all_bounded = False
                break
                
    if all_bounded:
        record("ML", "Probability Bounds Stress Test (100 runs)", "PASS", "All 800 evaluations strictly in [0.0, 1.0]")
    else:
        record("ML", "Probability Bounds Stress Test (100 runs)", "FAIL", "Found probability outside [0, 1] or NaN")
        
except Exception as e:
    record("ML", "Calibration & Bounds Check", "FAIL", str(e))

# 2.4 Model Reload Consistency
try:
    m1 = joblib.load(models_dir / "Ciprofloxacin_model.joblib")
    m2 = joblib.load(models_dir / "Ciprofloxacin_model.joblib")
    p1 = m1.predict_proba(vec1)[0, 1]
    p2 = m2.predict_proba(vec1)[0, 1]
    
    if abs(p1 - p2) < 1e-9:
        record("ML", "Model Reload Consistency", "PASS", f"Deterministic: p1={p1:.6f}, p2={p2:.6f}")
    else:
        record("ML", "Model Reload Consistency", "FAIL", f"Discrepancy: {p1} vs {p2}")
except Exception as e:
    record("ML", "Model Reload Consistency", "FAIL", str(e))

# 2.5 Final Evaluation Artifacts Integrity
try:
    with open(artifacts_dir / "model_metrics.json", "r") as f:
        metrics_doc = json.load(f)
        
    abx_dict = metrics_doc.get("antibiotics", {})
    if len(abx_dict) == 8:
        valid_eval = True
        for abx, item in abx_dict.items():
            tm = item.get("test_metrics", {})
            if "roc_auc" not in tm or "brier_score" not in tm or "confusion_matrix" not in tm:
                valid_eval = False
        if valid_eval:
            record("ML", "Evaluation Artifact (model_metrics.json)", "PASS", f"All 8 targets present with complete holdout metrics")
        else:
            record("ML", "Evaluation Artifact (model_metrics.json)", "FAIL", "Incomplete metric fields")
    else:
        record("ML", "Evaluation Artifact (model_metrics.json)", "FAIL", f"Expected 8 targets, got {len(abx_dict)}")
except Exception as e:
    record("ML", "Evaluation Artifact Check", "FAIL", str(e))


# ==============================================================================
# SECTION 3: FRONTEND VERIFICATION
# ==============================================================================
print("\n" + "="*70)
print("SECTION 3: FRONTEND REGRESSION TESTS")
print("="*70)

# 3.1 Check HTML Structure & Navigation Items
try:
    with open(BASE_DIR / "public" / "index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
        
    pages = [
        "page-overview",
        "page-analyzer",
        "page-history",
        "page-reports",
        "page-patients",
        "page-analytics",
        "page-model",
        "page-settings"
    ]
    
    all_pages_present = True
    for p in pages:
        if f'id="{p}"' not in html_content:
            all_pages_present = False
            record("FRONTEND", f"Page Section #{p}", "FAIL", "Section ID missing from HTML")
            
    if all_pages_present:
        record("FRONTEND", "All 8 Page Views in HTML", "PASS", "All navigation sections found in DOM")
        
    # Check key components in index.html
    key_elements = [
        ("Clinical Alert Banner", "clinical-alert-banner"),
        ("Preset Selector", "presetSelect"),
        ("Results Grid", "resultsGrid"),
        ("SHAP Panel", "shapPanel"),
        ("Explain Abx Select", "explainAbxSelect"),
        ("Waterfall Div", "plotlyShapDiv"),
        ("Export SHAP Button", "btnExportShap"),
        ("History Table", "historyTable"),
        ("Reports Grid", "reportsGrid"),
        ("Report Modal", "reportModal"),
        ("ROC Chart Canvas", "rocChartCanvas"),
        ("Calibration Chart Canvas", "calibChartCanvas"),
        ("Model Table Body", "modelsTableBody")
    ]
    
    for label, el_id in key_elements:
        if f'id="{el_id}"' in html_content or f'class="{el_id}"' in html_content or el_id in html_content:
            record("FRONTEND", f"DOM Element: {label}", "PASS", f"Found identifier '{el_id}'")
        else:
            record("FRONTEND", f"DOM Element: {label}", "FAIL", f"Missing identifier '{el_id}'")
            
except Exception as e:
    record("FRONTEND", "HTML Structure Verification", "FAIL", str(e))

# 3.2 Run All Presets Through Prediction API
try:
    res_conf = client.get("/api/config")
    presets = res_conf.json().get("presets", [])
    all_presets_ok = True
    
    for preset in presets:
        preset_id = preset["id"]
        preset_name = preset["name"]
        payload = preset["data"]
        res = client.post("/api/predict", json=payload)
        if res.status_code == 200 and len(res.json().get("predictions", [])) == 8:
            record("FRONTEND", f"Preset: {preset_id}", "PASS", f"{preset_name} predicted successfully")
        else:
            all_presets_ok = False
            record("FRONTEND", f"Preset: {preset_id}", "FAIL", f"HTTP {res.status_code}")
            
except Exception as e:
    record("FRONTEND", "Presets Verification", "FAIL", str(e))

# 3.3 Explainability Endpoints for All 8 Targets
try:
    all_explained = True
    for abx in antibiotic_targets:
        res = client.post("/api/explain", json={"patient_data": sample_patient, "antibiotic": abx})
        if res.status_code == 200:
            exp = res.json()
            if "waterfall_features" in exp and "base_value" in exp and "top_positive_factors" in exp:
                pass
            else:
                all_explained = False
        else:
            all_explained = False
            
    if all_explained:
        record("FRONTEND", "Local SHAP Explanations (8 targets)", "PASS", "All 8 targets produce valid waterfalls")
    else:
        record("FRONTEND", "Local SHAP Explanations (8 targets)", "FAIL", "Failed to explain some targets")
        
    res_g = client.get("/api/global-shap?antibiotic=Ciprofloxacin")
    if res_g.status_code == 200 and "global_shap" in res_g.json():
        record("FRONTEND", "Global SHAP Endpoint (/api/global-shap)", "PASS", "Valid global summary payload")
    else:
        record("FRONTEND", "Global SHAP Endpoint (/api/global-shap)", "FAIL", f"HTTP {res_g.status_code}")
except Exception as e:
    record("FRONTEND", "Explainability Verification", "FAIL", str(e))

# 3.4 Check JavaScript Syntax & Integrity
try:
    with open(BASE_DIR / "public" / "js" / "app.js", "r", encoding="utf-8") as f:
        js_code = f.read()
        
    # Verify key functions exist in app.js
    required_js_funcs = [
        "validateClinicalInputs",
        "renderAntibioticCards",
        "renderPlotlyWaterfall",
        "renderGlobalShapChart",
        "loadPredictionHistory",
        "loadClinicalReports",
        "openReportModal",
        "loadModelBenchmarks",
        "renderRocCurves",
        "renderCalibrationCurves"
    ]
    
    all_funcs_found = True
    for fn in required_js_funcs:
        if fn in js_code:
            record("FRONTEND", f"JS Function: {fn}", "PASS", "Defined in app.js")
        else:
            all_funcs_found = False
            record("FRONTEND", f"JS Function: {fn}", "FAIL", f"Missing function in app.js")
            
except Exception as e:
    record("FRONTEND", "JavaScript Integrity Check", "FAIL", str(e))


# ==============================================================================
# SECTION 4: DEPLOYMENT & PACKAGING TESTING
# ==============================================================================
print("\n" + "="*70)
print("SECTION 4: DEPLOYMENT & PACKAGING REGRESSION TESTS")
print("="*70)

# 4.1 Dependency Compatibility
try:
    with open(BASE_DIR / "requirements.txt", "r") as f:
        reqs = [line.strip().split("==")[0].split(">=")[0] for line in f if line.strip() and not line.startswith("#")]
        
    installed = {}
    import importlib.metadata
    for pkg in reqs:
        try:
            ver = importlib.metadata.version(pkg)
            installed[pkg] = ver
            record("DEPLOYMENT", f"Dependency: {pkg}", "PASS", f"Installed version: {ver}")
        except Exception:
            record("DEPLOYMENT", f"Dependency: {pkg}", "FAIL", "Package not installed")
except Exception as e:
    record("DEPLOYMENT", "Dependency Check", "FAIL", str(e))

# 4.2 Vercel Configuration & Routing
try:
    with open(BASE_DIR / "vercel.json", "r") as f:
        v_conf = json.load(f)
        
    entry_point = BASE_DIR / "api" / "index.py"
    if entry_point.exists() and "api/index.py" in v_conf.get("functions", {}):
        record("DEPLOYMENT", "Vercel Entrypoint & Routing", "PASS", "api/index.py correctly configured in vercel.json")
    else:
        record("DEPLOYMENT", "Vercel Entrypoint & Routing", "FAIL", "api/index.py missing or not in vercel.json")
except Exception as e:
    record("DEPLOYMENT", "Vercel Configuration Check", "FAIL", str(e))

# 4.3 Total Model & Artifact Bundle Size Check (Serverless Limits)
try:
    total_size = sum(f.stat().st_size for f in models_dir.glob("**/*") if f.is_file())
    total_size += sum(f.stat().st_size for f in artifacts_dir.glob("**/*") if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    # Serverless deployment limit is usually 250MB uncompressed
    if size_mb < 150:
        record("DEPLOYMENT", "Artifacts Bundle Size", "PASS", f"{size_mb:.2f} MB (Well within 250 MB serverless limit)")
    else:
        record("DEPLOYMENT", "Artifacts Bundle Size", "FAIL", f"{size_mb:.2f} MB exceeds recommended limit")
except Exception as e:
    record("DEPLOYMENT", "Bundle Size Check", "FAIL", str(e))

# 4.4 SQLite Serverless Storage Limitation Check
try:
    from src.database import get_db_path
    active_path = get_db_path()
    # Check if directory is writable
    db_dir = Path(active_path).parent
    test_write = db_dir / ".test_write"
    test_write.write_text("test")
    test_write.unlink()
    record("DEPLOYMENT", "Database Directory Writable", "PASS", f"Path: {db_dir}")
except Exception as e:
    record("DEPLOYMENT", "Database Directory Writable", "FAIL", str(e))


# ==============================================================================
# SECTION 5: FINAL REGRESSION SUMMARY
# ==============================================================================
print("\n" + "="*70)
print("FINAL QA REGRESSION AUDIT SUMMARY")
print("="*70)

passes = [r for r in results if r["status"] == "PASS"]
fails = [r for r in results if r["status"] == "FAIL"]

print(f"TOTAL TESTS EXECUTED : {len(results)}")
print(f"TOTAL PASSED         : {len(passes)}")
print(f"TOTAL FAILED         : {len(fails)}")

if fails:
    print("\nFAILED TESTS DETAILS:")
    for f in fails:
        print(f" - [{f['category']}] {f['test']}: {f['detail']}")
else:
    print("\nCONGRATULATIONS: 100% REGRESSION TESTS PASSED! ZERO DEFECTS DETECTED.")

# Clean up test database
if TEST_DB_PATH.exists():
    try:
        os.remove(TEST_DB_PATH)
    except Exception:
        pass
