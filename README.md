# Placement Prediction and Skill Gap Analysis Using Machine Learning

An end-to-end ML project that predicts a student's placement outcome and
identifies personalized skill gaps, based on the project proposal guided by
**Mr. Sandip Gupta**.

## 📌 Problem Statement
Students often can't tell whether they're placement-ready or which skills to
improve. This project uses ML to predict placement likelihood from academic
and skill-related data, and pinpoints the specific areas a student should
focus on — with a working prototype dashboard.

## 🗂️ Project Structure
```
placement_project/
├── data/
│   ├── placement_data.csv          # Dataset (1500 students, 16 columns incl. subject sub-scores)
│   └── progress.db                 # Per-student saved-assessment history (generated, SQLite)
├── config/
│   └── company_criteria.json       # Editable, illustrative company-eligibility rules (generated)
├── src/
│   ├── generate_dataset.py         # Creates the dataset (incl. DSA/DBMS/OS-CN, Quant/Logical/Verbal)
│   ├── preprocessing.py            # Cleaning, encoding, train/test split
│   ├── eda.py                      # Exploratory Data Analysis + plots
│   ├── train_models.py             # CV + tuning + calibration + uncertainty ensemble + versioning
│   ├── skill_gap.py                # Skill Gap Analysis engine (+ subject-wise & role-wise breakdown)
│   ├── skills_catalog.py           # Branch-wise named-skill catalog, scoring, and resource links
│   ├── job_roles.py                # Job-role skill catalog + role-readiness scoring
│   ├── company_eligibility.py      # Illustrative company-eligibility checker
│   └── progress_store.py           # SQLite-backed per-student progress history
├── notebooks/
│   └── Placement_Prediction_and_Skill_Gap_Analysis.ipynb   # Full walkthrough
├── app/
│   └── app.py                      # Streamlit dashboard (6 tabs — see below)
├── models/                         # Saved model + scaler + encoder (generated)
│   └── versions/<timestamp>/       # Versioned snapshot from every training run
├── reports/
│   ├── figures/                    # EDA, confusion-matrix, calibration & feature-importance charts
│   ├── model_comparison.csv        # Tuned test-set metrics per model
│   └── cv_comparison.csv           # 5-fold cross-validation F1 per model
├── tests/                          # pytest unit tests for the core logic modules
├── requirements.txt
└── README.md
```

## ⚙️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset
```bash
python src/generate_dataset.py
```
> Note: A real placement dataset (e.g., from Kaggle or your institution) can
> be dropped into `data/placement_data.csv` in place of the generated one —
> just keep the same column names, or update `preprocessing.py` accordingly.

### 3. Run EDA (optional, saves charts to `reports/figures/`)
```bash
python src/eda.py
```

### 4. Train & compare models
```bash
cd src
python train_models.py
```
This trains **Logistic Regression, Decision Tree, Random Forest, and
XGBoost** with class-imbalance handling, 5-fold cross-validation, and a
small hyperparameter search each; evaluates every tuned model with
Accuracy / Precision / Recall / F1-score + confusion matrix; calibrates the
winning model's probabilities; trains a separate Random Forest "uncertainty
ensemble" for confidence bands; and saves a timestamped version snapshot
under `models/versions/`.

### 5. Run the tests (optional)
```bash
pytest tests/ -q
```

### 6. Launch the dashboard
```bash
streamlit run app/app.py
```

### 7. Explore the full walkthrough
Open `notebooks/Placement_Prediction_and_Skill_Gap_Analysis.ipynb` in
Jupyter for the narrated pipeline (EDA → modeling → skill gap).

## 🖥️ Dashboard Tabs

1. **🎯 Predict** — enter a student's profile, get a calibrated placement
   probability with a confidence band, a percentile comparison against
   branch-mates, the skill-gap report (with a subject-wise DSA/DBMS/OS-CN
   and Quant/Logical/Verbal breakdown), named-skill recommendations with
   learning-resource links, a SHAP explainability chart showing *why* the
   model predicted what it did, an illustrative company-eligibility check,
   a downloadable PDF report, and a button to save the run to your progress
   history.
2. **🔧 What-If** — pick one factor (internships, CGPA, technical score, …)
   and see a live curve of how the placement probability would change if
   you improved just that one thing, holding everything else fixed.
3. **📁 Batch** — upload a CSV for an entire class/cohort and get
   predictions + an "at risk" shortlist, downloadable as CSV.
4. **📈 My Progress** — enter a Student ID (a label, not a real login) to
   see a trend line of saved assessments over time.
5. **📊 Model Insights** — tuned model comparison, 5-fold CV results, class
   balance, calibration curve, and EDA charts.
6. **ℹ️ About** — project & methodology overview.

## 🧠 Methodology

1. **Data Collection** — Academic performance, aptitude (with Quant /
   Logical / Verbal sub-scores), technical skills (with DSA / DBMS / OS-CN
   sub-scores), communication skills, projects, internships, certifications,
   backlogs, branch, and placement status.
2. **Preprocessing** — Missing-value handling, deduplication, label encoding
   for the categorical `Branch` feature, feature scaling.
3. **EDA** — Class balance, correlation heatmap, CGPA/skills vs placement,
   placement rate by branch.
4. **Modeling** — Logistic Regression, Decision Tree, Random Forest,
   XGBoost — each cross-validated (5-fold) and hyperparameter-tuned via
   `GridSearchCV`; class imbalance handled via `class_weight='balanced'` /
   `scale_pos_weight`.
5. **Calibration & Uncertainty** — the best model (by test-F1) is wrapped in
   `CalibratedClassifierCV` so displayed probabilities are trustworthy; a
   separate Random Forest ensemble provides a tree-vote confidence band.
6. **Evaluation** — Accuracy, Precision, Recall, F1-score, Confusion Matrix,
   calibration curve.
7. **Skill Gap Analysis** — For a given student, benchmarks their skills
   against the average profile of *successfully placed* students in the same
   branch, weighted by each feature's importance (from the trained model),
   with subject-level breakdowns, and returns a prioritized, plain-language
   improvement plan with resource links.
7b. **Job-Role Skill Gap** — The student also picks a **target job role**
   (e.g. Data Scientist, DevOps Engineer, Embedded Systems Engineer — 19
   roles across all 6 branches, see `src/job_roles.py`). Their selected
   skills are matched against that role's specific weighted skill catalog to
   produce a **Role Readiness %** and a ranked "skills to learn next for
   this role" list with resource links — independent of, and in addition
   to, the branch-wide benchmark above.
8. **Explainability** — SHAP (`TreeExplainer` / `LinearExplainer` depending
   on the winning model) shows per-feature contribution for each prediction.
9. **Deployment** — Streamlit dashboard: single prediction, what-if
   simulator, batch scoring, progress tracking, illustrative company
   eligibility, and PDF export.

## 📊 Sample Results
See `reports/model_comparison.csv`, `reports/cv_comparison.csv`, and the
charts in `reports/figures/` after running `train_models.py`. On the
generated dataset, model performance sits in the 0.55–0.65 F1 range —
expected, since the data is intentionally generated with realistic
overlap/noise between placed and non-placed students rather than being
artificially easy to separate. Swapping in a real dataset will change these
numbers.

## ⚠️ Important Caveats
- The dataset is **simulated**, not real institutional data.
- The company-eligibility list in `config/company_criteria.json` is
  **illustrative** — it is not a live feed of real recruiter criteria. Edit
  it with your placement cell's actual thresholds before treating it as
  more than a demo.
- `progress_store.py`'s "Student ID" is a label for separating saved
  histories on one shared deployment, **not** an authentication system —
  there's no password or identity check.

## 📚 Suggested References
- Research papers on placement prediction / student performance prediction using ML
- Scikit-learn, XGBoost, SHAP, and Streamlit official documentation
- Kaggle placement-prediction datasets (for a real-data variant)

## 🔮 Further Ideas
- Replace the simulated dataset with a real placement dataset once available.
- Real-time company eligibility feed via placement-cell API/integration.
- Multi-user authentication if deployed beyond a single classroom/lab.
- SHAP summary (global) plot in the Model Insights tab across the whole test set.
