"""
generate_dataset.py
--------------------
Generates a realistic synthetic dataset for the Placement Prediction &
Skill Gap Analysis project. Since curated Kaggle placement datasets rarely
contain all the skill-level attributes needed for a proper skill-gap
module, this script builds a statistically realistic dataset (1500 students)
using controlled random distributions + a logical placement rule with noise,
mimicking how such data behaves in the real world.

Subject-wise granularity
-------------------------
On top of the original coarse features (Technical_Skills, Aptitude_Score),
this version also generates subject-level sub-scores:
  - Technical_Skills is now the average of DSA_Score, DBMS_Score, OS_CN_Score
  - Aptitude_Score is now the average of Quant_Score, Logical_Score, Verbal_Score
These sub-scores are kept as extra columns in the CSV. The trained ML model
still only consumes the aggregate columns (so existing model artifacts /
app code that only knows about Technical_Skills & Aptitude_Score keep
working), but the Skill Gap module uses the sub-scores to give students
subject-by-subject recommendations instead of one vague "improve technical
skills" message.

Run:
    python src/generate_dataset.py
Output:
    data/placement_data.csv
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

N = 1500

# ---- Core attributes ----
cgpa = np.clip(np.random.normal(7.2, 0.9, N), 5.0, 10.0).round(2)

# Subject-wise aptitude sub-scores (correlated with each other, with noise)
apt_base = np.random.normal(65, 14, N)
quant_score = np.clip(apt_base + np.random.normal(0, 8, N), 15, 100).round(1)
logical_score = np.clip(apt_base + np.random.normal(0, 8, N), 15, 100).round(1)
verbal_score = np.clip(apt_base + np.random.normal(0, 10, N), 15, 100).round(1)
aptitude_score = ((quant_score + logical_score + verbal_score) / 3).round(1)

# Subject-wise technical sub-scores
tech_base = np.random.normal(60, 17, N)
dsa_score = np.clip(tech_base + np.random.normal(0, 10, N), 5, 100).round(1)
dbms_score = np.clip(tech_base + np.random.normal(0, 10, N), 5, 100).round(1)
os_cn_score = np.clip(tech_base + np.random.normal(0, 10, N), 5, 100).round(1)
technical_skills = ((dsa_score + dbms_score + os_cn_score) / 3).round(1)

communication_skills = np.clip(np.random.normal(62, 15, N), 10, 100).round(1)
projects_count = np.clip(np.random.poisson(2.5, N), 0, 8)
internships_count = np.clip(np.random.poisson(1.0, N), 0, 4)
certifications_count = np.clip(np.random.poisson(1.8, N), 0, 6)
backlogs = np.clip(np.random.poisson(0.4, N), 0, 6)

branches = np.random.choice(
    ["CSE", "IT", "ECE", "EEE", "MECH", "CIVIL"],
    size=N, p=[0.30, 0.20, 0.20, 0.10, 0.12, 0.08]
)

# ---- Latent "placement score" (drives probability of placement) ----
score = (
    0.9 * (cgpa - 5) / 5
    + 0.8 * (aptitude_score / 100)
    + 1.1 * (technical_skills / 100)
    + 0.7 * (communication_skills / 100)
    + 0.25 * projects_count
    + 0.5 * internships_count
    + 0.2 * certifications_count
    - 0.6 * backlogs
)

# add noise so it isn't a trivial linear-separable problem
score += np.random.normal(0, 0.6, N)

# convert to probability via logistic function, then sample outcome
prob = 1 / (1 + np.exp(-(score - score.mean())))
placement_status = np.random.binomial(1, prob)

df = pd.DataFrame({
    "CGPA": cgpa,
    "Aptitude_Score": aptitude_score,
    "Quant_Score": quant_score,
    "Logical_Score": logical_score,
    "Verbal_Score": verbal_score,
    "Technical_Skills": technical_skills,
    "DSA_Score": dsa_score,
    "DBMS_Score": dbms_score,
    "OS_CN_Score": os_cn_score,
    "Communication_Skills": communication_skills,
    "Projects_Count": projects_count,
    "Internships_Count": internships_count,
    "Certifications_Count": certifications_count,
    "Backlogs": backlogs,
    "Branch": branches,
    "Placement_Status": placement_status,  # 1 = Placed, 0 = Not Placed
})

out_path = os.path.join(DATA_DIR, "placement_data.csv")
df.to_csv(out_path, index=False)
print(f"Saved {out_path} with shape {df.shape}")
print(df["Placement_Status"].value_counts(normalize=True))
