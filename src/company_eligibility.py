"""
company_eligibility.py
-----------------------
Checks which companies (from an editable, illustrative criteria file) a
student currently meets the minimum eligibility bar for, based on common
recruiter screening filters: minimum CGPA, minimum technical/aptitude
score, maximum active backlogs, and (optionally) eligible branches.

IMPORTANT: The company list and thresholds shipped here are illustrative
placeholders, NOT real, current recruiter data. There is no public feed of
live company eligibility criteria, so this module is designed to be edited
by you (or read from `config/company_criteria.json`) with your institution's
actual placement-cell data before being presented to students as real
guidance.
"""

import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "company_criteria.json")

DEFAULT_CRITERIA = [
    {
        "company": "Example Corp A (Mass Recruiter)",
        "min_cgpa": 6.0,
        "min_technical_skills": 40,
        "min_aptitude_score": 40,
        "max_backlogs": 1,
        "branches": ["CSE", "IT", "ECE", "EEE", "MECH", "CIVIL"],
    },
    {
        "company": "Example Corp B (IT Services)",
        "min_cgpa": 6.5,
        "min_technical_skills": 45,
        "min_aptitude_score": 50,
        "max_backlogs": 0,
        "branches": ["CSE", "IT", "ECE"],
    },
    {
        "company": "Example Corp C (Product / Core Tech)",
        "min_cgpa": 7.5,
        "min_technical_skills": 70,
        "min_aptitude_score": 65,
        "max_backlogs": 0,
        "branches": ["CSE", "IT"],
    },
    {
        "company": "Example Corp D (Core Engineering)",
        "min_cgpa": 6.5,
        "min_technical_skills": 50,
        "min_aptitude_score": 45,
        "max_backlogs": 0,
        "branches": ["ECE", "EEE", "MECH", "CIVIL"],
    },
]


def load_criteria() -> list:
    """Load company criteria from config/company_criteria.json if present,
    otherwise fall back to (and create) the illustrative default list."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(DEFAULT_CRITERIA, f, indent=2)
    return DEFAULT_CRITERIA


def check_eligibility(student: dict, branch: str) -> list:
    """
    student: dict with CGPA, Technical_Skills, Aptitude_Score, Backlogs
    branch: student's branch

    Returns a list of dicts: {company, eligible: bool, reasons: [str]}
    """
    criteria = load_criteria()
    results = []
    for c in criteria:
        reasons = []
        if branch not in c.get("branches", []):
            reasons.append(f"Not currently hiring from {branch}")
        if student.get("CGPA", 0) < c["min_cgpa"]:
            reasons.append(f"CGPA below {c['min_cgpa']}")
        if student.get("Technical_Skills", 0) < c["min_technical_skills"]:
            reasons.append(f"Technical score below {c['min_technical_skills']}")
        if student.get("Aptitude_Score", 0) < c["min_aptitude_score"]:
            reasons.append(f"Aptitude score below {c['min_aptitude_score']}")
        if student.get("Backlogs", 0) > c["max_backlogs"]:
            reasons.append(f"More than {c['max_backlogs']} active backlog(s)")

        results.append({
            "company": c["company"],
            "eligible": len(reasons) == 0,
            "reasons": reasons,
        })
    return results


if __name__ == "__main__":
    sample = {"CGPA": 7.2, "Technical_Skills": 55, "Aptitude_Score": 60, "Backlogs": 0}
    for r in check_eligibility(sample, "CSE"):
        status = "✅ Eligible" if r["eligible"] else "❌ Not yet — " + "; ".join(r["reasons"])
        print(f"{r['company']}: {status}")
