# AMR-X: Explainable AI Antimicrobial Resistance Decision Support

[![Live Prototype](https://img.shields.io/badge/Live_Prototype-amr--chi.vercel.app-0D9488?style=for-the-badge&logo=vercel)](https://amr-chi.vercel.app/)
[![Python](https://img.shields.io/badge/Python-3.10_|_3.11_|_3.12-1E3A8A?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn_&_XGBoost-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/XAI-SHAP_TreeExplainer-7C3AED?style=for-the-badge)](https://shap.readthedocs.io)
[![Tests](https://img.shields.io/badge/Tests-45_Passed-059669?style=for-the-badge&logo=pytest)](https://pytest.org)

**Active Prototype URL:** [https://amr-chi.vercel.app/](https://amr-chi.vercel.app/)  
**Classification:** Research & Algorithmic Prototype (Synthetic Cohort Decision Support)  
**System Architecture:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | **Model Card:** [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | **Competition Showcase Guide:** [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md)

---

> [!WARNING]
> **Essential Clinical & Regulatory Disclaimer:**  
> AMR-X is an educational and algorithmic **research prototype** trained on a synthetically generated cohort ($N=20,000$). It is **NOT** a medical device and has **NOT** undergone clinical validation or regulatory review (FDA/CE). The system does **NOT** provide medication prescriptions, dosage recommendations, or substitute for professional clinical judgment. Microbiological culture isolation and standard phenotypic antimicrobial susceptibility testing (AST) remain strictly mandatory before antimicrobial administration.

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Project Objectives](#2-project-objectives)
3. [System Architecture](#3-system-architecture)
4. [Dataset Source & Synthetic Nature](#4-dataset-source--synthetic-nature)
5. [Sixteen Input Features](#5-sixteen-input-features)
6. [Eight Antibiotic Targets](#6-eight-antibiotic-targets)
7. [Preprocessing Methodology](#7-preprocessing-methodology)
8. [Data Splitting Methodology](#8-data-splitting-methodology)
9. [Random Forest & XGBoost Implementation](#9-random-forest--xgboost-implementation)
10. [Model Selection Tournament](#10-model-selection-tournament)
11. [Evaluation Metrics & Holdout Benchmark](#11-evaluation-metrics--holdout-benchmark)
12. [Explainable AI Methodology (SHAP)](#12-explainable-ai-methodology-shap)
13. [FastAPI Backend](#13-fastapi-backend)
14. [Database Design & Concurrency](#14-database-design--concurrency)
15. [Clinical Dashboard Modules](#15-clinical-dashboard-modules)
16. [Installation & Execution](#16-installation--execution)
17. [API Endpoints Reference](#17-api-endpoints-reference)
18. [Known Scientific Limitations](#18-known-scientific-limitations)
19. [Future Scope](#19-future-scope)

---

## 1. Problem Statement
Antimicrobial resistance (AMR) is one of the most critical threats facing modern healthcare. When a patient presents with an acute infection, phenotypic antimicrobial susceptibility testing (AST) typically requires 24 to 48 hours to isolate the pathogen and determine minimum inhibitory concentrations (MIC). During this diagnostic delay, clinicians are forced to prescribe empiric antimicrobial therapy based on aggregate population antibiograms. This frequently leads to two adverse failure modes:
1. Prescribing an ineffective narrow-spectrum agent that fails due to unrecognized resistance.
2. Over-prescribing broad-spectrum "last-line" antibiotics (e.g., carbapenems, fluoroquinolones), accelerating hospital-wide resistance selection pressure.

AMR-X addresses this empiric window by leveraging tabular machine learning to estimate patient-specific resistance risks across 8 common therapeutic options using available presentation data (demographics, laboratory biomarkers, infection presentation, and prior clinical exposures).

---

## 2. Project Objectives
- **Predict Non-Susceptibility Probabilities:** Simultaneously estimate resistance probabilities for 8 key therapeutic antibiotics.
- **Use 16 Standard Clinical Features:** Constrain inputs strictly to routinely available presentation parameters without requiring genomic sequencing.
- **Empirical Algorithm Tournament:** Independently compare Random Forest and XGBoost per target and automatically deploy the superior estimator.
- **Holdout Probability Calibration:** Calibrate raw tree scores using Platt sigmoid scaling on an independent partition to ensure probabilities reflect true empirical frequencies.
- **Mathematically Defensible Explainability:** Implement SHAP `TreeExplainer` with verified additivity to surface local patient feature attributions and global cohort drivers.
- **Reliable Data Persistence & Reporting:** Archive all inference events in a transactional SQLite store and generate standardized vector PDF antibiogram reports.

---

## 3. System Architecture

```mermaid
flowchart TD
    subgraph Client["Presentation Tier (Browser Dashboard)"]
        UI["Clinical Intelligence UI (HTML5 / Vanilla CSS / ES6)"]
        Analyzer["16-Feature Patient Analyzer with Presets"]
        Plots["Attribution & Curve Engine (Plotly.js + Chart.js)"]
    end

    subgraph Server["API Gateway Tier (FastAPI)"]
        API["FastAPI Controller (app/main.py)"]
        Schema["Pydantic v2 Schemas (Input Validation)"]
    end

    subgraph Pipeline["ML & XAI Engine"]
        Pre["Preprocessor (StandardScaler + OneHotEncoder)"]
        Tourney["Selected Ensembles (4 Random Forest + 4 XGBoost)"]
        Calib["Platt Sigmoid Calibrators (N=2,400 Holdout Fit)"]
        SHAP["SHAP TreeExplainer Engine"]
    end

    subgraph Persistence["Storage & Document Generation"]
        DB[(SQLite WAL Database: data/amr_history.db)]
        PDF["ReportLab PDF Antibiogram Generator"]
    end

    UI -->|POST /api/predict| Schema
    Schema --> API
    API --> Pre
    Pre --> Tourney
    Tourney --> Calib
    Calib --> API
    API -->|Insert Prediction Record| DB
    API -->|Return Calibrated Antibiogram| UI

    UI -->|POST /api/explain| API
    API --> SHAP
    SHAP --> Plots

    UI -->|GET /api/reports/{id}/pdf| API
    API --> PDF
    PDF --> UI
```

---

## 4. Dataset Source & Synthetic Nature
- **Cohort File:** `data/amr_synthetic_dataset.csv`
- **Total Records:** 20,000 synthetic patient infection episodes.
- **Dataset Nature:** The dataset is a synthetic research cohort generated to simulate realistic correlations between patient comorbidities, healthcare exposures, and antimicrobial non-susceptibility. It contains no Protected Health Information (PHI).
- **Target Prevalence Distribution:**
  - *Fosfomycin:* 10.2%
  - *Nitrofurantoin:* 21.9%
  - *Gentamicin:* 29.5%
  - *Ceftriaxone:* 41.5%
  - *Trimethoprim-Sulfamethoxazole:* 53.6%
  - *Levofloxacin:* 56.4%
  - *Amoxicillin-Clavulanate:* 57.8%
  - *Ciprofloxacin:* 61.5%

---

## 5. Sixteen Input Features

The predictive feature space is constrained to 16 parameters grouped into 4 clinical sections:

### 1. Demographics & Patient Baseline
- `age`: Chronological age in years (Valid: 18–110).
- `sex`: Biological sex at birth (`F` or `M`).

### 2. Laboratory Biomarkers
- `creatinine_mg_dl`: Serum creatinine level in mg/dL (Valid: 0.1–25.0; Normal ref: 0.70–1.30 mg/dL).
- `wbc_count_k_ul`: White blood cell count in k/µL (Valid: 0.1–100.0; Normal ref: 4.5–11.0 k/µL).

### 3. Infection Presentation & Microbiology
- `infection_type`: Syndrome category (`UTI`, `Catheter_Associated_UTI`, `Complicated_UTI`, `Pyelonephritis`).
- `organism`: Isolated microbial species (`E_coli`, `Klebsiella_pneumoniae`, `Enterococcus_faecalis`, `Proteus_mirabilis`, `Pseudomonas_aeruginosa`, `Staph_saprophyticus`).
- `num_prior_uti_1yr`: Documented UTI recurrences within prior 12 months (0–50).

### 4. Clinical History & Exposure Flags (Binary 0/1)
- `recent_hospitalization_90d`: Inpatient hospital admission within the last 90 days.
- `recent_antibiotic_use_90d`: Systemic antimicrobial exposure within the last 90 days.
- `prior_resistant_culture_1yr`: Historical microbiological isolate exhibiting antimicrobial resistance.
- `diabetes`: Documented Type 1 or Type 2 Diabetes Mellitus.
- `catheter_use`: Indwelling urinary catheter present at time of culture.
- `immunocompromised`: Active immunosuppression, chemotherapy, or biological therapy.
- `nursing_home_resident`: Residence in a skilled nursing or long-term care facility.
- `travel_last_6mo`: History of international travel within the preceding 6 months.
- `healthcare_worker`: Occupational clinical healthcare environment exposure.

### Strict Leakage Exclusion
- `patient_id` is unconditionally removed from the feature matrix to prevent patient-level memorization.
- All target-derived synthetic columns (`*_resistance_probability`) are stripped upon dataset loading.

---

## 6. Eight Antibiotic Targets
AMR-X models non-susceptibility across 8 critical first-line and second-line antimicrobial agents:
1. **Nitrofurantoin** (Target: `Nitrofurantoin_resistant`)
2. **Trimethoprim-Sulfamethoxazole (TMP-SMX)** (Target: `Trimethoprim-Sulfamethoxazole_resistant`)
3. **Ciprofloxacin** (Target: `Ciprofloxacin_resistant`)
4. **Levofloxacin** (Target: `Levofloxacin_resistant`)
5. **Amoxicillin-Clavulanate** (Target: `Amoxicillin-Clavulanate_resistant`)
6. **Ceftriaxone** (Target: `Ceftriaxone_resistant`)
7. **Fosfomycin** (Target: `Fosfomycin_resistant`)
8. **Gentamicin** (Target: `Gentamicin_resistant`)

---

## 7. Preprocessing Methodology
Feature transformations are encapsulated in a Scikit-Learn `ColumnTransformer` (`artifacts/preprocessor.joblib`):
- **Numeric Features (4):** Scaled via `StandardScaler(with_mean=True, with_std=True)`.
- **Categorical Features (3):** Encoded via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- **Binary Clinical Flags (9):** Passed through as verified binary integer features.
- **Strict Leakage Isolation:** The preprocessor is fitted **strictly on the 56% training split** ($N=11,200$) and never sees validation, calibration, or test data during fitting.

---

## 8. Data Splitting Methodology
Each target antibiotic undergoes an independent, reproducible 4-way stratified partition (fixed seed `42`):

```
Dataset (N = 20,000)
├── 1. Training Split (56%, N = 11,200)      ──> Fits Preprocessor & Candidate Classifiers
├── 2. Validation Split (12%, N = 2,400)    ──> Selects Winning Model (RF vs XGBoost)
├── 3. Calibration Split (12%, N = 2,400)   ──> Fits Platt Sigmoid Calibrator (CalibratedClassifierCV)
└── 4. Untouched Test Split (20%, N = 4,000) ──> Evaluates Final Benchmarks (artifacts/model_metrics.json)
```

The test split is held out untouched; zero test records or labels are ever used for model tuning, preprocessor fitting, or probability calibration.

---

## 9. Random Forest & XGBoost Implementation
For every antibiotic target, two distinct non-linear architectures are trained:
- **Random Forest (`RandomForestClassifier`):**
  - Ensembles of 100–150 decision trees.
  - Subsampling with bootstrap aggregation and `class_weight='balanced'` to penalize minority resistant misclassifications.
  - Max tree depth constrained to 8–12 to prevent overfitting.
- **XGBoost (`XGBClassifier`):**
  - Gradient-boosted decision trees with `max_depth=3–5`, `learning_rate=0.05–0.1`.
  - `scale_pos_weight` set dynamically based on training split class prevalence to handle class imbalance.
  - Subsample ratio $0.80$, colsample by tree $0.80$.

---

## 10. Model Selection Tournament
Models are evaluated on the held-out validation split ($N=2,400$). Because prescribing an ineffective antibiotic due to a false negative is clinically hazardous, the selection objective prioritizes resistant-class recall and discriminative capacity over overall accuracy:

$$\text{Selection Score} = 0.45 \times \text{ROC-AUC} + 0.35 \times \text{F1-Score} + 0.20 \times \text{Recall}_{\text{resistant}}$$

### Tournament Outcome (4 RF vs. 4 XGBoost)
The objective tournament selected a balanced 4-to-4 distribution:
- **Random Forest Winners (4):** Trimethoprim-Sulfamethoxazole, Ciprofloxacin, Levofloxacin, Amoxicillin-Clavulanate.
- **XGBoost Winners (4):** Nitrofurantoin, Ceftriaxone, Fosfomycin, Gentamicin.

---

## 11. Evaluation Metrics & Holdout Benchmark

All performance metrics are evaluated strictly on the untouched holdout test partition ($N=4,000$, 20%) and stored in `artifacts/model_metrics.json`:

| Antibiotic Target | Winning Model | Holdout ROC-AUC | Holdout F1-Score | Resistant Recall | Precision | Accuracy | Brier Calibration Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Nitrofurantoin** | XGBoost | 0.634 | 0.014 | 0.007 | 0.500 | 0.781 | **0.164** |
| **Trimethoprim-Sulfamethoxazole** | Random Forest | 0.675 | 0.677 | 0.706 | 0.651 | 0.633 | **0.225** |
| **Ciprofloxacin** | Random Forest | 0.675 | 0.746 | 0.819 | 0.685 | 0.657 | **0.216** |
| **Levofloxacin** | Random Forest | 0.675 | 0.704 | 0.759 | 0.656 | 0.636 | **0.223** |
| **Amoxicillin-Clavulanate** | Random Forest | 0.658 | 0.720 | 0.800 | 0.655 | 0.632 | **0.224** |
| **Ceftriaxone** | XGBoost | 0.698 | 0.593 | 0.542 | 0.654 | 0.658 | **0.219** |
| **Fosfomycin** | XGBoost | 0.601 | 0.000 | 0.000 | 0.000 | 0.898 | **0.091** |
| **Gentamicin** | XGBoost | 0.630 | 0.223 | 0.139 | 0.566 | 0.701 | **0.203** |

*Scientific Note on Low-Prevalence Targets:* For Fosfomycin (prevalence 10.2%), standard unweighted thresholding yields high accuracy ($89.8\%$) and tight Brier calibration ($0.091$), but zero unweighted recall at $\tau = 0.50$. In AMR-X, continuous calibrated probabilities and tiered risk bands ($<30\%$, $30\text{--}59\%$, $\ge 60\%$) prevent discrete threshold failure modes.

---

## 12. Explainable AI Methodology (SHAP)
- **Local Explanations:** For any patient inference, SHAP `TreeExplainer` decomposes the prediction into additive feature attributions ($\phi_i$).
  - For Random Forest models: Values represent probability shifts ($\Delta p$).
  - For XGBoost models: Values represent margin log-odds shifts.
- **Mathematical Additivity:** The engine validates that $f(x) = E[f(x)] + \sum_{i=1}^M \phi_i$ within $1\times 10^{-4}$ tolerance.
- **Global Explanations:** Aggregates mean absolute SHAP values across the cohort to highlight population-level risk drivers.
- **Correlation vs. Causation Guardrail:** The UI and reports explicitly state that SHAP attributions reflect mathematical model weights and do *not* establish biological or clinical causality.

---

## 13. FastAPI Backend
The API backend (`app/main.py`) provides high-throughput, low-latency asynchronous decision support:
- **Pydantic v2 Validation:** Strict input bounds checking (rejecting unphysiological ages, negative laboratory values, and malformed categories).
- **Latency:** Average end-to-end inference latency under $200\text{ ms}$ for all 8 antibiotic predictions.
- **Defensive Error Handling:** Structured error payloads (RFC 7807 format) with stack trace suppression.
- **Static Asset Serving:** Built-in mounting of static files (`/static`) with cache headers.

---

## 14. Database Design & Concurrency
- **Engine:** SQLite 3 with Write-Ahead Logging (`PRAGMA journal_mode=WAL`).
- **Connection Safety:** Context-managed transactions (`src/database.py`) supporting automatic commit/rollback.
- **Configurable Storage:** Database location is environment-configurable via `AMR_DB_PATH` to guarantee total test isolation.
- **Tables:**
  - `predictions`: Primary table storing unique `prediction_id`, timestamp, patient profile, and 8 individual antibiotic probabilities.
  - `patients`: Index of distinct patient encounters and longitudinal risk history.

---

## 15. Clinical Dashboard Modules
1. **Overview Dashboard:** Live database statistics (total analyses, distinct patients, high-risk cases), recent activity feed, and clinical stewardship alert banner.
2. **Patient Analyzer:** 4-section clinical input form with built-in reference ranges, demographic presets, and client-side validation.
3. **Prediction Results:** 8 antibiotic probability cards with 30% and 60% threshold markers, model badges, and one-click "Explain Decision" routing.
4. **Explainable AI (SHAP):** Interactive Plotly waterfall chart, top 5 positive/negative factors, global feature importance toggle, and JSON export.
5. **Prediction History:** Chronological SQLite audit log with copyable prediction IDs, live search, and risk-tier filtering.
6. **Clinical Reports:** Antibiogram report cards, modal preview, print stylesheet, and direct ReportLab PDF download (`/api/reports/{id}/pdf`).
7. **Model Intelligence:** Architecture comparison cards, full holdout benchmark table, ROC curves, and reliability/calibration curves.

---

## 16. Installation & Execution

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Standard virtual environment tool (`venv`)

### Setup Instructions
```bash
# 1. Clone repository & enter directory
git clone https://github.com/<your-org>/AMR-main.git
cd AMR-main/AMR-main

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running Automated Tests
Execute the complete test suite (45 tests covering ML pipelines, calibration, SHAP additivity, API validation, database persistence, and PDF reports):
```bash
python -m pytest tests -v
```

### Retraining ML Models (Optional)
To execute the complete 4-way stratified pipeline, model selection, calibration, and benchmark generation from scratch:
```bash
python src/train.py
```

### Launching the Web Server
Start the local FastAPI ASGI server:
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Access the application at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

---

## 17. API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the single-page clinical dashboard |
| `GET` | `/api/config` | System configuration, feature bounds, and clinical presets |
| `POST` | `/api/predict` | Computes calibrated resistance probabilities across 8 antibiotics & archives to DB |
| `POST` | `/api/explain` | Computes local SHAP waterfall attribution for a selected antibiotic |
| `GET` | `/api/global-shap` | Retrieves precomputed global feature importances for a selected antibiotic |
| `GET` | `/api/models` | Retrieves model registry, architecture selection metadata, and test metrics |
| `GET` | `/api/stats` | Retrieves live database summary metrics (total analyses, distinct patients) |
| `GET` | `/api/history` | Queries historical prediction audit logs with search and risk filters |
| `GET` | `/api/history/{id}` | Retrieves full inputs, outputs, and metadata for a specific prediction ID |
| `GET` | `/api/patients` | Retrieves directory of distinct patient profiles and historical risk trajectory |
| `GET` | `/api/reports` | Lists generated standardized clinical antibiogram reports |
| `GET` | `/api/reports/{id}` | Retrieves structured JSON antibiogram report for preview |
| `GET` | `/api/reports/{id}/pdf` | Generates and streams an authentic multi-page vector PDF report (ReportLab) |
| `GET` | `/api/eda` | Returns cohort summary statistics (pathogen distributions, syndrome prevalence) |

---

## 18. Known Scientific Limitations
1. **Synthetic Data Substrate:** Trained exclusively on simulated data. Real-world microbiological dynamics, local hospital antibiograms, and patient physiological variations will differ significantly.
2. **Extreme Class Imbalance:** Low-prevalence antibiotics (e.g., Fosfomycin at 10.2%) produce low unweighted recall at discrete thresholds without cost-sensitive adjustment.
3. **Point-in-Time Prediction:** Predictions reflect baseline presentation risk and do not incorporate longitudinal treatment response or evolving culture data.
4. **Absence of Pharmacokinetic (PK/PD) Modeling:** Probabilities model in vitro non-susceptibility and do not account for drug penetration, renal clearance, or patient weight.

---

## 19. Future Scope
- **Prospective Clinical Validation:** Partnering with clinical microbiology labs to evaluate model calibration on real-world de-identified EHR cohorts.
- **Genomic & Proteomic Feature Integration:** Extending the feature schema to incorporate rapid whole-genome sequencing (WGS) or PCR resistance gene markers (e.g., *bla*<sub>KPC</sub>, *mecA*).
- **FHIR / EHR Interoperability:** Integrating SMART on FHIR protocols to ingest lab values and infection notes directly from Epic and Cerner systems.
- **Adaptive Antibiogram Transfer Learning:** Enabling hospitals to fine-tune pre-trained models on local institutional antibiograms using low-rank adaptation.

---

## License & Attribution
This project is licensed under the MIT License for academic, competition, and research exploration.  
**Live Showcase Prototype:** [https://amr-chi.vercel.app/](https://amr-chi.vercel.app/)
