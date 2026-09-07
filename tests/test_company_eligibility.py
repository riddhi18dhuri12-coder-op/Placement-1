import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from company_eligibility import check_eligibility, load_criteria  # noqa: E402

STRONG_STUDENT = {"CGPA": 9.0, "Technical_Skills": 90, "Aptitude_Score": 90, "Backlogs": 0}
WEAK_STUDENT = {"CGPA": 5.0, "Technical_Skills": 10, "Aptitude_Score": 10, "Backlogs": 3}


def test_load_criteria_returns_nonempty_list():
    criteria = load_criteria()
    assert isinstance(criteria, list)
    assert len(criteria) > 0
    for c in criteria:
        assert "company" in c and "min_cgpa" in c


def test_strong_student_eligible_for_more_companies_than_weak_student():
    strong_results = check_eligibility(STRONG_STUDENT, "CSE")
    weak_results = check_eligibility(WEAK_STUDENT, "CSE")
    strong_eligible = sum(r["eligible"] for r in strong_results)
    weak_eligible = sum(r["eligible"] for r in weak_results)
    assert strong_eligible >= weak_eligible


def test_weak_student_gets_reasons_for_ineligibility():
    results = check_eligibility(WEAK_STUDENT, "CSE")
    for r in results:
        if not r["eligible"]:
            assert len(r["reasons"]) > 0


def test_branch_not_in_list_is_flagged():
    results = check_eligibility(STRONG_STUDENT, "ARTS")
    branch_flagged = any(
        any("hiring from" in reason for reason in r["reasons"])
        for r in results
    )
    assert branch_flagged
