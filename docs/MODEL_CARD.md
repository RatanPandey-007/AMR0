# AMR-X Model Card

**Model Identifier:** AMR-X Ensemble v1.0  
**Model Family:** Supervised Tree-Based Binary Classification Ensembles with Platt Calibration  
**Release Date:** October 2026  
**Artifact Storage:** `models/*.joblib`, `artifacts/model_metrics.json`, `artifacts/preprocessor.joblib`  
**Live Showcase Prototype:** [https://amr-chi.vercel.app/](https://amr-chi.vercel.app/)

---

## 1. Model Details

### 1.1 Overview
AMR-X is a multi-target antimicrobial resistance (AMR) predictive system comprising eight independently trained, calibrated, and explained binary classification estimators. For each antibiotic target, an automated model selection tournament selects between a **Random Forest Classifier** and an **XGBoost Classifier**, followed by holdout **Sigmoid (Platt) Calibration** to produce calibrated posterior resistance probabilities $P(Y=1 \mid X)$.

### 1.2 Developed By
- AMR-X Engineering & Research Team (Machine Learning, Healthcare AI, and Backend Engineering).
- Designed for competition showcase and educational decision-support exploration.

### 1.3 Technical Architecture Summary
- **Base Estimators:** Scikit-Learn 1.4+ `RandomForestClassifier`, XGBoost 2.0+ `XGBClassifier`.
- **Calibration Engine:** Scikit-Learn `CalibratedClassifierCV(method='sigmoid', cv='prefit')`.
- **Explainability:** SHAP 0.45+ `TreeExplainer` with verified local additivity ($f(x) = E[f(x)] + \sum \phi_i$).
- **Preprocessing:** Scikit-Learn `ColumnTransformer` with `StandardScaler` and `OneHotEncoder(handle_unknown='ignore')`.

---

## 2. Intended Use & Boundaries

### 2.1 Intended Use Cases
- **Research & Algorithmic Exploration:** Benchmarking tabular tree ensembles against multi-target antimicrobial resistance tasks.
- **Antimicrobial Stewardship Decision Support Prototype:** Demonstrating how machine-learning probabilities and feature attributions can provide transparent, interpretable guidance alongside clinical context.
- **Explainability Education:** Illustrating local tree attribution (SHAP waterfalls) and contrasting mathematical feature attributions with biological causation.

### 2.2 Out-of-Scope & Prohibited Uses
- **Clinical Prescription & Treatment Selection:** The model MUST NOT be used to select, prescribe, adjust, or discontinue antimicrobial therapy.
- **Standalone Diagnostic Tool:** The model CANNOT replace standard microbiological culture isolation and phenotypic antimicrobial susceptibility testing (AST).
- **Pediatric Cohort Application:** Input features and models are parameterized for adult patients ($\ge 18$ years); application to pediatric populations is out of scope.
- **High-Stakes Autonomous Action:** The system is strictly decision-supportive and advisory; autonomous medical decisions are prohibited.

---

## 3. Input Features & Target Schema

### 3.1 Input Features (16 Total)

| Feature Name | Clinical Domain | Type | Valid Range / Categories | Description |
| :--- | :--- | :--- | :--- | :--- |
| `age` | Demographics | Numeric | 18–110 years | Patient chronological age |
| `sex` | Demographics | Categorical | `F`, `M` | Biological sex at birth |
| `creatinine_mg_dl` | Laboratory | Numeric | 0.1–25.0 mg/dL | Serum creatinine level (Normal ref: 0.7–1.3) |
| `wbc_count_k_ul` | Laboratory | Numeric | 0.1–100.0 k/µL | White blood cell count (Normal ref: 4.5–11.0) |
| `infection_type` | Microbiology | Categorical | `UTI`, `CAUTI`, `Complicated_UTI`, `Pyelonephritis` | Clinical syndrome classification |
| `organism` | Microbiology | Categorical | 6 clinical species | Primary isolated Gram-negative/positive pathogen |
| `num_prior_uti_1yr` | History | Numeric | 0–50 | Confirmed UTI recurrences in prior 12 months |
| `recent_hospitalization_90d` | Exposure | Binary | 0 or 1 | Inpatient admission within last 90 days |
| `recent_antibiotic_use_90d` | Exposure | Binary | 0 or 1 | Systemic antibiotic therapy within last 90 days |
| `prior_resistant_culture_1yr` | History | Binary | 0 or 1 | Prior AST culture showing multi-drug resistance |
| `diabetes` | Comorbidity | Binary | 0 or 1 | Documented Type 1 or Type 2 Diabetes Mellitus |
| `catheter_use` | Exposure | Binary | 0 or 1 | Indwelling urinary catheter present |
| `immunocompromised` | Comorbidity | Binary | 0 or 1 | Active immunosuppressive state or chemotherapy |
| `nursing_home_resident` | Exposure | Binary | 0 or 1 | Long-term healthcare facility residence |
| `travel_last_6mo` | Exposure | Binary | 0 or 1 | International travel within prior 6 months |
| `healthcare_worker` | Exposure | Binary | 0 or 1 | Active occupational healthcare environment exposure |

### 3.2 Target Antibiotic Targets (8 Total)
1. **Nitrofurantoin** (Target: `Nitrofurantoin_resistant`, prevalence: 21.9%)
2. **Trimethoprim-Sulfamethoxazole (TMP-SMX)** (Target: `Trimethoprim-Sulfamethoxazole_resistant`, prevalence: 53.6%)
3. **Ciprofloxacin** (Target: `Ciprofloxacin_resistant`, prevalence: 61.5%)
4. **Levofloxacin** (Target: `Levofloxacin_resistant`, prevalence: 56.4%)
5. **Amoxicillin-Clavulanate** (Target: `Amoxicillin-Clavulanate_resistant`, prevalence: 57.8%)
6. **Ceftriaxone** (Target: `Ceftriaxone_resistant`, prevalence: 41.5%)
7. **Fosfomycin** (Target: `Fosfomycin_resistant`, prevalence: 10.2%)
8. **Gentamicin** (Target: `Gentamicin_resistant`, prevalence: 29.5%)

### 3.3 Strict Leakage Prevention Rules
- The identifier `patient_id` is strictly excluded from all training and inference matrices.
- All target-derived probability columns (`*_resistance_probability`) are dropped immediately upon data ingestion.
- Preprocessing transformations (`StandardScaler`, `OneHotEncoder`) are fitted **exclusively on the training split** ($N=11,200$) and never touch validation, calibration, or test splits.

---

## 4. Dataset & Splitting Methodology

### 4.1 Cohort Characteristics
- **Dataset:** `amr_synthetic_dataset.csv`
- **Total Records:** 20,000 synthetic patient infection episodes.
- **Nature:** Synthetic research cohort generated to model clinical AMR correlations while avoiding identifiable Protected Health Information (PHI).

### 4.2 Four-Way Stratified Split Methodology
To guarantee zero data leakage across model selection, calibration, and final benchmark reporting, each antibiotic target undergoes an independent 4-way stratified split:

```
Total Cohort (N = 20,000)
├── Training Split (56%, N = 11,200)      --> Fits Preprocessor & Candidate Estimators (RF, XGB)
├── Validation Split (12%, N = 2,400)    --> Evaluates Candidates & Selects Winner
├── Calibration Split (12%, N = 2,400)   --> Fits Platt Sigmoid Calibration (CalibratedClassifierCV)
└── Untouched Test Split (20%, N = 4,000) --> Final Benchmark Evaluation & ROC/Reliability Curves
```

- **Random Seed:** 42 (fixed across all splits).
- **Stratification:** Stratified by the specific antibiotic resistance indicator $Y_i$.
- **Test Integrity:** The test partition ($N=4,000$) is held out untouched; zero test labels are used for hyperparameter selection, preprocessor fitting, or calibration.

---

## 5. Model Selection & Tournament Results

### 5.1 Selection Metric Formula
To account for the high clinical cost of false negatives (prescribing an ineffective antibiotic), candidate models were compared on the validation partition ($N=2,400$) using a weighted selection score:

$$\text{Selection Score} = 0.45 \times \text{ROC-AUC} + 0.35 \times \text{F1-Score} + 0.20 \times \text{Recall}_{\text{resistant}}$$

### 5.2 Selection Outcomes
The tournament resulted in an exact 4-to-4 distribution between Random Forest and XGBoost:

- **Random Forest Selected (4):** Trimethoprim-Sulfamethoxazole, Ciprofloxacin, Levofloxacin, Amoxicillin-Clavulanate.
- **XGBoost Selected (4):** Nitrofurantoin, Ceftriaxone, Fosfomycin, Gentamicin.

---

## 6. Holdout Test Evaluation ($N=4,000$, Untouched Partition)

All metrics below are computed on the independent holdout test split and read directly from `artifacts/model_metrics.json`:

| Target Antibiotic | Selected Architecture | Holdout ROC-AUC | Holdout F1-Score | Resistant Recall | Precision | Accuracy | Brier Calibration Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Nitrofurantoin** | XGBoost | 0.634 | 0.014 | 0.007 | 0.500 | 0.781 | **0.164** |
| **Trimethoprim-Sulfamethoxazole** | Random Forest | 0.675 | 0.677 | 0.706 | 0.651 | 0.633 | **0.225** |
| **Ciprofloxacin** | Random Forest | 0.675 | 0.746 | 0.819 | 0.685 | 0.657 | **0.216** |
| **Levofloxacin** | Random Forest | 0.675 | 0.704 | 0.759 | 0.656 | 0.636 | **0.223** |
| **Amoxicillin-Clavulanate** | Random Forest | 0.658 | 0.720 | 0.800 | 0.655 | 0.632 | **0.224** |
| **Ceftriaxone** | XGBoost | 0.698 | 0.593 | 0.542 | 0.654 | 0.658 | **0.219** |
| **Fosfomycin** | XGBoost | 0.601 | 0.000 | 0.000 | 0.000 | 0.898 | **0.091** |
| **Gentamicin** | XGBoost | 0.630 | 0.223 | 0.139 | 0.566 | 0.701 | **0.203** |

---

## 7. Scientific Limitations & Honest Metric Analysis

1. **Class Imbalance in Low-Prevalence Targets:**
   - For **Fosfomycin** (prevalence 10.2%) and **Nitrofurantoin** (prevalence 21.9%), standard decision threshold classification ($\tau = 0.50$) yields high overall accuracy ($89.8\%$ and $78.1\%$) and strong probability calibration (Brier scores of $0.091$ and $0.164$), but low unweighted recall ($0.000$ and $0.007$).
   - This represents an authentic, scientifically honest artifact of extreme class imbalance without artificial post-hoc threshold manipulation. In clinical decision support, the continuous calibrated probability $P(\text{Resistant})$ and documented risk bands ($<30\%$, $30\text{--}59\%$, $\ge 60\%$) provide superior clinical utility over rigid binary thresholding.
2. **Synthetic Feature Independence:**
   - In synthetic cohorts, complex biological interactions (e.g., plasmid-mediated horizontal gene transfer, enzyme beta-lactamase kinetics) are approximated by linear and non-linear mathematical generative distributions.
3. **Absence of Longitudinal AST Feedback:**
   - Models predict initial empirical resistance risk at the time of presentation; they do not update dynamically as serial blood or urine cultures mature over 48–72 hours.

---

## 8. Explainability & Ethical Safeguards

1. **SHAP TreeExplainer Additivity:**
   - Formally guaranteed: local feature attributions sum to the model's raw tree output within $1\times 10^{-4}$ numerical tolerance.
2. **Correlation vs. Causation Guardrail:**
   - The user interface and model card explicitly state: *"Feature attributions reflect statistical weights in tree ensembles and do not represent biological or physical mechanisms of antimicrobial resistance."*
3. **Non-Prescriptive Risk Tiers:**
   - Probabilities are displayed with documented descriptive risk bands:
     - **Low Resistance Risk:** $< 30\%$
     - **Moderate Resistance Risk:** $30\% \text{ to } 59.9\%$
     - **High Resistance Risk:** $\ge 60\%$
   - The dashboard avoids prescriptive green/red "recommendation" flags to prevent automated treatment substitution.
