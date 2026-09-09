"""
skill_gap.py
------------
Skill Gap Analysis module.

Logic:
1. Build a "successful profile" = mean value of each skill-related feature
   among students who WERE placed (Placement_Status == 1), computed
   per-branch so comparisons stay fair across disciplines.
2. For a given student's inputs, compute the gap between their value and
   the successful-profile value for each feature.
3. Weight each gap by that feature's importance (from the trained model's
   feature_importances_, falling back to equal weights if unavailable) to
   produce a prioritized list of the areas most worth improving.
4. Translate gaps into plain-language, actionable recommendations.
5. If subject-level sub-scores are present in the dataset (DSA/DBMS/OS_CN
   for technical, Quant/Logical/Verbal for aptitude), also produce a
   subject-by-subject breakdown instead of one vague "Technical_Skills"
   or "Aptitude_Score" line.
"""

import os
import pandas as pd
from skills_catalog import get_missing_skills
from job_roles import get_missing_role_skills, compute_role_readiness

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "placement_data.csv")
IMPORTANCE_PATH = os.path.join(BASE_DIR, "models", "feature_importances.csv")

SKILL_FEATURES = [
    "CGPA", "Aptitude_Score", "Technical_Skills", "Communication_Skills",
    "Projects_Count", "Internships_Count", "Certifications_Count",
]
# Backlogs is inverse (lower is better) so it's handled separately.

TECHNICAL_SUBSCORES = ["DSA_Score", "DBMS_Score", "OS_CN_Score"]
APTITUDE_SUBSCORES = ["Quant_Score", "Logical_Score", "Verbal_Score"]

RECOMMENDATIONS = {
    "CGPA": "Focus on improving core-subject grades; many recruiters use CGPA as an initial screening cutoff.",
    "Aptitude_Score": "Practice quantitative aptitude, logical reasoning and verbal ability daily — these are the most common first-round filters.",
    "Technical_Skills": "Strengthen data structures, algorithms and your core branch subjects; practice on coding platforms regularly.",
    "Communication_Skills": "Work on spoken English, group discussions and mock interviews to build confidence and clarity.",
    "Projects_Count": "Build 1-2 more solid, resume-worthy projects (ideally with real-world relevance and documentation).",
    "Internships_Count": "Try to secure at least one internship — even a short one — to gain practical, resume-boosting experience.",
    "Certifications_Count": "Complete relevant certifications (e.g., in your specialization or trending tech stacks) to validate your skills.",
    "DSA_Score": "Drill Data Structures & Algorithms daily on a coding platform (arrays, trees, graphs, DP) — this is the single most common technical interview filter.",
    "DBMS_Score": "Revise DBMS fundamentals: normalization, SQL joins/queries, transactions and indexing — a frequent core-subject interview topic.",
    "OS_CN_Score": "Revise Operating Systems (processes, threads, memory, scheduling) and Computer Networks (OSI/TCP-IP, DNS, HTTP) basics.",
    "Quant_Score": "Practice quantitative aptitude daily (arithmetic, percentages, time-speed-distance, data interpretation).",
    "Logical_Score": "Practice logical reasoning puzzles and pattern-based questions — timed practice sets help the most.",
    "Verbal_Score": "Build verbal ability through reading comprehension, grammar and vocabulary practice.",
}


def _has_subscores(df: pd.DataFrame) -> tuple:
    has_tech = all(c in df.columns for c in TECHNICAL_SUBSCORES)
    has_apt = all(c in df.columns for c in APTITUDE_SUBSCORES)
    return has_tech, has_apt


def build_successful_profile(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Mean feature values among placed students, grouped by branch."""
    placed = df[df["Placement_Status"] == 1]
    profile = placed.groupby("Branch")[features].mean()
    overall = placed[features].mean()
    return profile, overall


def load_feature_weights():
    if os.path.exists(IMPORTANCE_PATH):
        imp = pd.read_csv(IMPORTANCE_PATH, index_col=0).squeeze("columns")
        weights = {feat: imp.get(feat, 0.1) for feat in SKILL_FEATURES}
    else:
        weights = {feat: 1.0 for feat in SKILL_FEATURES}
    # normalize
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


def _gap_row(feat, student_val, benchmark_val, weight):
    raw_gap = benchmark_val - student_val
    pct_gap = (raw_gap / benchmark_val * 100) if benchmark_val != 0 else 0
    weighted_score = max(pct_gap, 0) * weight
    return {
        "Feature": feat,
        "Your_Value": student_val,
        "Benchmark_Value": round(benchmark_val, 2),
        "Gap_Percent": round(pct_gap, 1),
        "Priority_Score": round(weighted_score, 3),
    }


def analyze_subject_gap(df: pd.DataFrame, branch: str, student: dict, subscore_cols: list) -> pd.DataFrame:
    """Subject-by-subject gap for a group of sub-scores (technical or aptitude)."""
    profile_by_branch, overall_profile = build_successful_profile(df, subscore_cols)
    target = profile_by_branch.loc[branch] if branch in profile_by_branch.index else overall_profile
    rows = []
    for feat in subscore_cols:
        student_val = student.get(feat)
        if student_val is None:
            continue
        rows.append(_gap_row(feat, student_val, target[feat], 1.0 / len(subscore_cols)))
    return pd.DataFrame(rows).sort_values("Priority_Score", ascending=False) if rows else pd.DataFrame()


def analyze_skill_gap(student: dict, branch: str = None, known_skills: list = None,
                       role: str = None) -> dict:
    """
    student: dict with keys matching SKILL_FEATURES (+ optional 'Backlogs',
        and optionally the subject sub-scores DSA_Score/DBMS_Score/OS_CN_Score
        and Quant_Score/Logical_Score/Verbal_Score for finer granularity).
    branch: optional branch name to compare against branch-specific profile
    known_skills: optional list of specific skill names the student selected
        (e.g. ['Python', 'SQL']) — used to recommend specific missing skills
        instead of just a generic "improve technical skills" message.
    role: optional target job role (e.g. "Data Scientist"). When provided,
        the specific-skill recommendation and "skills to learn next" list
        are driven by what THAT ROLE needs (via job_roles.py) instead of
        the generic branch-wide catalog, and a role-readiness score is
        added to the result.

    Returns a dict with per-feature gap, a priority-ranked list, human
    -readable recommendations for the top gaps, and (when available)
    subject-wise sub-score breakdowns for technical & aptitude skills.
    """
    df = pd.read_csv(DATA_PATH)
    has_tech_sub, has_apt_sub = _has_subscores(df)

    profile_by_branch, overall_profile = build_successful_profile(df, SKILL_FEATURES)

    if branch and branch in profile_by_branch.index:
        target_profile = profile_by_branch.loc[branch]
    else:
        target_profile = overall_profile

    weights = load_feature_weights()

    gaps = [
        _gap_row(feat, student.get(feat, 0), target_profile[feat], weights.get(feat, 0.1))
        for feat in SKILL_FEATURES
    ]
    gaps_df = pd.DataFrame(gaps).sort_values("Priority_Score", ascending=False)

    top_gaps = gaps_df[gaps_df["Gap_Percent"] > 0].head(3)
    recommendations = []
    for _, row in top_gaps.iterrows():
        feat = row["Feature"]
        if feat == "Technical_Skills" and role and known_skills is not None:
            # A target role was given — recommend skills specific to THAT
            # role rather than the generic branch-wide catalog.
            missing = get_missing_role_skills(role, known_skills, top_n=3)
            if missing:
                skill_names = ", ".join(m["Skill"] for m in missing)
                advice = (f"Your technical skill set is below the benchmark for placed students, and "
                          f"below what recruiters typically look for in a {role}. Consider learning: "
                          f"{skill_names}.")
            else:
                advice = RECOMMENDATIONS.get(feat, "Work on improving this area.")
        elif feat == "Technical_Skills" and branch and known_skills is not None:
            # Swap the generic technical-skills message for specific,
            # named skills the student is missing.
            missing = get_missing_skills(branch, known_skills, top_n=3)
            if missing:
                skill_names = ", ".join(m["Skill"] for m in missing)
                advice = (f"Your technical skill set is below the benchmark for {branch} students who "
                          f"got placed. Consider learning: {skill_names}.")
            else:
                advice = RECOMMENDATIONS.get(feat, "Work on improving this area.")
        else:
            advice = RECOMMENDATIONS.get(feat, "Work on improving this area.")
        recommendations.append({"Area": feat, "Advice": advice})

    result = {
        "gap_table": gaps_df.reset_index(drop=True),
        "recommendations": recommendations,
    }

    # Always surface the top missing skills separately too, even if
    # Technical_Skills wasn't in the top-3 gaps — useful as a standalone
    # "skills to add next" checklist in the UI.
    if branch and known_skills is not None:
        result["missing_skills"] = get_missing_skills(branch, known_skills, top_n=5)

    # ---- Role-specific skill gap (only when a target role was given) ----
    if role and known_skills is not None:
        result["role"] = role
        result["role_readiness"] = compute_role_readiness(role, known_skills)
        result["missing_role_skills"] = get_missing_role_skills(role, known_skills, top_n=6)

    # ---- Subject-wise breakdown (if the dataset/student input has it) ----
    if has_tech_sub and all(c in student for c in TECHNICAL_SUBSCORES):
        tech_df = analyze_subject_gap(df, branch, student, TECHNICAL_SUBSCORES)
        result["technical_subject_gap"] = tech_df.reset_index(drop=True)
        if not tech_df.empty:
            worst = tech_df.iloc[0]
            if worst["Gap_Percent"] > 0:
                result.setdefault("subject_recommendations", []).append({
                    "Area": worst["Feature"],
                    "Advice": RECOMMENDATIONS.get(worst["Feature"], "Work on improving this subject."),
                })

    if has_apt_sub and all(c in student for c in APTITUDE_SUBSCORES):
        apt_df = analyze_subject_gap(df, branch, student, APTITUDE_SUBSCORES)
        result["aptitude_subject_gap"] = apt_df.reset_index(drop=True)
        if not apt_df.empty:
            worst = apt_df.iloc[0]
            if worst["Gap_Percent"] > 0:
                result.setdefault("subject_recommendations", []).append({
                    "Area": worst["Feature"],
                    "Advice": RECOMMENDATIONS.get(worst["Feature"], "Work on improving this subject."),
                })

    return result


if __name__ == "__main__":
    # quick manual test
    sample_student = {
        "CGPA": 6.8,
        "Aptitude_Score": 55,
        "Technical_Skills": 45,
        "Communication_Skills": 50,
        "Projects_Count": 1,
        "Internships_Count": 0,
        "Certifications_Count": 1,
    }
    result = analyze_skill_gap(sample_student, branch="CSE")
    print(result["gap_table"])
    print("\nTop Recommendations:")
    for r in result["recommendations"]:
        print(f"- {r['Area']}: {r['Advice']}")
