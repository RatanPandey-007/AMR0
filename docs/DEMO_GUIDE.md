# AMR-X Competition Showcase & Demonstration Guide

**Project Name:** AMR-X (Clinical AI Antimicrobial Resistance Decision Support)  
**Live Prototype:** [https://amr-chi.vercel.app/](https://amr-chi.vercel.app/)  
**Target Audience:** Competition Judges, Healthcare AI Researchers, Technical Reviewers

---

## 1. Quick Start & Execution

### Option A: Live Cloud Prototype (Recommended)
Access the live deployed prototype directly in your browser:  
👉 **[https://amr-chi.vercel.app/](https://amr-chi.vercel.app/)**

### Option B: Local Execution (FastAPI Backend)
To run the full stack locally with active SQLite persistence and PDF generation:

```bash
# 1. Clone or open repository
cd AMR-main

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Run the complete automated test suite
python -m pytest tests -v

# 4. Launch the application
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in any modern web browser.

---

## 2. Five-Minute Showcase Demonstration Script

### Step 1: Institutional Overview & Stewardship Banner (0:00 – 0:45)
1. **Navigate to "Overview"** in the sidebar.
2. **Highlight the Research Prototype Banner:** Point out the top amber alert banner. Emphasize that AMR-X explicitly disclaims production clinical readiness and highlights the synthetic cohort origin.
3. **Show Live Database Telemetry:** Note the live counters reading directly from the SQLite database (`/api/stats`):
   - *Total Historical Analyses*
   - *Distinct Patients Profiled*
   - *High-Resistance Cases Identified*
   - *8 Calibrated ML Estimators Online*
4. **Point to Stewardship Core:** Review the dark navy antimicrobial stewardship mandate emphasizing narrow-spectrum empiric conservation.

---

### Step 2: Patient Analyzer & Input Validation (0:45 – 1:45)
1. **Navigate to "Patient Analyzer"** in the sidebar.
2. **Review the 4 Logical Feature Sections:**
   - *Demographics & Patient Baseline:* Age (`18–110 yrs`), biological sex.
   - *Laboratory Biomarkers:* Creatinine (with clinical ref `0.70–1.30 mg/dL`), WBC (with clinical ref `4.5–11.0 k/µL`).
   - *Infection Presentation & Microbiology:* Infection syndrome, isolated organism, recurrence count.
   - *Clinical History & Exposure Flags:* 9 high-contrast toggle cards for healthcare exposures.
3. **Demonstrate Input Validation Safeguards:**
   - Clear the Age input or set it to `-5` and click **Analyze Resistance Risk**.
   - Show how the client-side validation immediately highlights the field with an accessible red border and displays a human-readable error without crashing.
4. **Load Clinical Preset:**
   - Select **"Elderly Complicated UTI / CAUTI"** from the Preset dropdown.
   - Note how all 16 clinical parameters populate with a realistic high-risk profile (Age 78, Creatinine 1.85 mg/dL, WBC 14.8 k/µL, *Klebsiella pneumoniae*, CAUTI, catheter present).
5. **Click "Analyze Resistance Risk":**
   - Note the real-time latency indicator ($\sim 180\text{ ms}$).

---

### Step 3: Calibrated Predictions & Risk Bands (1:45 – 2:30)
1. **Inspect the 8 Antibiotic Risk Cards:**
   - Note that probabilities are presented across all 8 therapeutic targets simultaneously.
   - Point out the **Calibrated Risk Bands**:
     - *Low Risk:* $< 30\%$
     - *Moderate Risk:* $30\% \text{ to } 59.9\%$
     - *High Risk:* $\ge 60\%$
2. **Explain Threshold Markers:**
   - Point to the horizontal progress tracks: lines at 30% and 60% mark the clinical boundaries clearly.
3. **Highlight Architectural Diversity:**
   - Notice that some targets use `XGBoost · Calibrated` while others use `Random Forest · Calibrated`.
4. **Trigger Explainability:**
   - On the highest-risk drug (e.g., Ciprofloxacin or Ceftriaxone), click the **"Explain Decision"** button.

---

### Step 4: Explainable AI & SHAP Waterfall (2:30 – 3:30)
1. **Review Local SHAP Waterfall Breakdown:**
   - Show the interactive Plotly waterfall chart decomposing the prediction from baseline $E[f(x)]$ to model output $f(x)$.
   - Point to the red increasing bars (risk factors like prior resistant culture, catheter use) vs. teal decreasing bars.
2. **Verify Additivity & Space:**
   - For Random Forest, attributions are displayed in probability space ($\Delta p$).
   - For XGBoost, attributions are displayed in log-odds margin space.
3. **Toggle Global Feature Importance:**
   - Click the **"Global Importance"** button to view dataset-wide mean $|SHAP|$ scores for that drug.
4. **Export SHAP Payload:**
   - Click **"Export SHAP"** to download the exact mathematical decomposition as a JSON file (`AMR_SHAP_<Antibiotic>.json`).
5. **Highlight Attribution vs. Causation Guardrail:**
   - Emphasize the disclaimer note: *Feature attributions reflect statistical model contributions, NOT clinical causality.*

---

### Step 5: Clinical Reports & PDF Download (3:30 – 4:15)
1. **Navigate to "Reports"** in the sidebar.
2. **Preview Report:**
   - Click **"Preview"** on the latest report card.
   - Review the standardized Antibiogram Intelligence Report displaying patient demographics, labs, isolated pathogen, highest estimated non-susceptibility callout, and full multi-agent matrix.
3. **Demonstrate Direct PDF Export:**
   - Click **"Download PDF"** in the modal.
   - Open the generated PDF: show the clean, vector-rendered ReportLab document complete with institutional header, complete table, and medical research disclaimer.

---

### Step 6: Model Intelligence & Benchmark Verification (4:15 – 5:00)
1. **Navigate to "Model Intelligence"** in the sidebar.
2. **Model Selection Highlights:**
   - Review the four comparison cards summarizing the 4-way stratified methodology ($N_{\text{train}}=11,200$, $N_{\text{val}}=2,400$, $N_{\text{calib}}=2,400$, $N_{\text{test}}=4,000$).
3. **Holdout Benchmark Table:**
   - Show the live benchmark table reading from `artifacts/model_metrics.json`.
   - Point out that **all 8 models report honest, unvarnished holdout metrics**.
   - Note the low-prevalence targets (Fosfomycin F1 $0.000$ due to class imbalance at $10.2\%$ prevalence) as evidence of genuine scientific integrity.
4. **ROC & Calibration Curves:**
   - Show the 8-curve ROC chart (all models exceeding the $AUC=0.500$ chance baseline).
   - Show the Reliability Curves demonstrating tight alignment with the perfect calibration diagonal ($y=x$).

---

## 3. Anticipated Judge Questions & Defensible Answers

### Q1: "How did you prevent data leakage during model training?"
> **Answer:** "We implemented strict architectural isolation across four phases:
> 1. We immediately dropped `patient_id` and all `*_resistance_probability` columns from the feature matrix.
> 2. We generated a 4-way stratified split per antibiotic: 56% Train ($N=11,200$), 12% Validation ($N=2,400$), 12% Calibration ($N=2,400$), and 20% Holdout Test ($N=4,000$).
> 3. The `StandardScaler` and `OneHotEncoder` preprocessor was fitted exclusively on the 56% training split.
> 4. Probability calibration (Platt sigmoid) was fitted exclusively on the 12% calibration split. The 20% holdout test split was kept strictly untouched until final metric evaluation."

---

### Q2: "Why do you use both Random Forest and XGBoost rather than just one algorithm?"
> **Answer:** "Different antimicrobial mechanisms exhibit different tabular interaction structures. Rather than assuming one model fits all 8 targets, we ran an empirical selection tournament on the validation partition ($N=2,400$) using a multi-metric score:
> $$\text{Selection Score} = 0.45 \times \text{ROC-AUC} + 0.35 \times \text{F1} + 0.20 \times \text{Recall}$$
> The tournament selected Random Forest for 4 targets (Ciprofloxacin, Levofloxacin, TMP-SMX, Amoxicillin-Clavulanate) and XGBoost for 4 targets (Nitrofurantoin, Ceftriaxone, Fosfomycin, Gentamicin)."

---

### Q3: "Why is the holdout F1-score for Fosfomycin 0.000?"
> **Answer:** "Fosfomycin has the lowest prevalence in the cohort at only 10.2%. Under standard unweighted thresholding ($\tau = 0.50$), the model optimizes accuracy (achieving 89.8% holdout accuracy) and achieves strong probability calibration (Brier score of 0.091), but predicts zero positive cases. Rather than manufacturing fake metrics or arbitrarily adjusting thresholds post-hoc, we report the authentic mathematical result. In our clinical dashboard, we present the continuous calibrated probability ($P \approx 10\%$) and documented risk tier rather than relying on binary classification."

---

### Q4: "Can this system be deployed in a hospital tomorrow?"
> **Answer:** "No, and we explicitly document this limitation across the application. AMR-X is a research and educational prototype trained on a synthetic dataset of 20,000 records. Before any clinical deployment, prospective observational clinical trials, laboratory microbiological validation, electronic health record (EHR/FHIR) integration, and regulatory review would be strictly required. AMR-X provides decision support, not prescriptive medical advice."
