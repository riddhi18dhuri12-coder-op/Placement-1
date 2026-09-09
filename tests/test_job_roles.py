import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from job_roles import (  # noqa: E402
    list_roles, ordered_roles_for_branch, get_role_catalog, get_combined_catalog,
    compute_role_readiness, get_missing_role_skills, get_role_resource_link,
)
from skill_gap import analyze_skill_gap  # noqa: E402


def test_list_roles_nonempty_and_unique():
    roles = list_roles()
    assert len(roles) > 10
    assert len(roles) == len(set(roles))


def test_ordered_roles_for_branch_puts_relevant_roles_first():
    ordered = ordered_roles_for_branch("CSE")
    assert ordered[0] in ("Software Development Engineer (Backend)", "Full Stack Developer",
                           "Frontend Developer", "Data Scientist", "Machine Learning Engineer",
                           "DevOps / Cloud Engineer", "Cybersecurity Analyst",
                           "QA / Test Automation Engineer", "Data Analyst", "Business Analyst")
    assert set(ordered) == set(list_roles())


def test_unknown_branch_still_returns_all_roles():
    ordered = ordered_roles_for_branch("NOT_A_BRANCH")
    assert set(ordered) == set(list_roles())


def test_role_catalog_nonempty_for_known_role():
    catalog = get_role_catalog("Data Scientist")
    assert "Python" in catalog
    assert "Machine Learning / AI" in catalog


def test_combined_catalog_merges_branch_and_role():
    combined = get_combined_catalog("CSE", "Data Scientist")
    assert "Data Structures & Algorithms" in combined  # from branch
    assert "Pandas / NumPy" in combined  # from role


def test_role_readiness_zero_with_no_skills():
    assert compute_role_readiness("Data Scientist", []) == 0.0


def test_role_readiness_increases_with_relevant_skills():
    low = compute_role_readiness("Data Scientist", ["Python"])
    high = compute_role_readiness("Data Scientist", list(get_role_catalog("Data Scientist").keys()))
    assert 0 < low < high
    assert high == 100.0


def test_missing_role_skills_excludes_known():
    missing = get_missing_role_skills("Data Scientist", ["Python"], top_n=10)
    assert all(m["Skill"] != "Python" for m in missing)


def test_role_resource_link_falls_back_gracefully():
    assert isinstance(get_role_resource_link("Totally Made Up Skill"), str)
    assert get_role_resource_link("Python") != ""


def test_analyze_skill_gap_with_role_adds_role_fields():
    student = {
        "CGPA": 6.8, "Aptitude_Score": 55, "Technical_Skills": 45, "Communication_Skills": 50,
        "Projects_Count": 1, "Internships_Count": 0, "Certifications_Count": 1,
    }
    result = analyze_skill_gap(student, branch="CSE", known_skills=["Python"], role="Data Scientist")
    assert result["role"] == "Data Scientist"
    assert 0 <= result["role_readiness"] <= 100
    assert "missing_role_skills" in result


def test_analyze_skill_gap_without_role_omits_role_fields():
    student = {
        "CGPA": 6.8, "Aptitude_Score": 55, "Technical_Skills": 45, "Communication_Skills": 50,
        "Projects_Count": 1, "Internships_Count": 0, "Certifications_Count": 1,
    }
    result = analyze_skill_gap(student, branch="CSE", known_skills=["Python"])
    assert "role" not in result
    assert "role_readiness" not in result
