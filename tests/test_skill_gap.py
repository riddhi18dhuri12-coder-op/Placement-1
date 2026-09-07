import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from skill_gap import analyze_skill_gap, SKILL_FEATURES  # noqa: E402

WEAK_STUDENT = {
    "CGPA": 5.5, "Aptitude_Score": 30, "Technical_Skills": 20, "Communication_Skills": 25,
    "Projects_Count": 0, "Internships_Count": 0, "Certifications_Count": 0,
}

STRONG_STUDENT = {
    "CGPA": 9.5, "Aptitude_Score": 98, "Technical_Skills": 98, "Communication_Skills": 95,
    "Projects_Count": 10, "Internships_Count": 5, "Certifications_Count": 8,
}


def test_gap_table_has_one_row_per_skill_feature():
    result = analyze_skill_gap(WEAK_STUDENT, branch="CSE")
    assert len(result["gap_table"]) == len(SKILL_FEATURES)
    assert set(result["gap_table"]["Feature"]) == set(SKILL_FEATURES)


def test_weak_student_has_positive_gaps_and_recommendations():
    result = analyze_skill_gap(WEAK_STUDENT, branch="CSE")
    assert (result["gap_table"]["Gap_Percent"] > 0).any()
    assert len(result["recommendations"]) > 0
    assert len(result["recommendations"]) <= 3


def test_strong_student_has_few_or_no_positive_gaps():
    result = analyze_skill_gap(STRONG_STUDENT, branch="CSE")
    positive_gaps = (result["gap_table"]["Gap_Percent"] > 0).sum()
    # A student far above the placed-student benchmark on everything
    # should have very few (ideally zero) meaningful gaps.
    assert positive_gaps <= 2


def test_missing_skills_present_when_known_skills_given():
    result = analyze_skill_gap(WEAK_STUDENT, branch="CSE", known_skills=["Python"])
    assert "missing_skills" in result
    assert len(result["missing_skills"]) > 0


def test_missing_skills_absent_when_known_skills_not_given():
    result = analyze_skill_gap(WEAK_STUDENT, branch="CSE")
    assert "missing_skills" not in result


def test_technical_recommendation_names_specific_skills():
    result = analyze_skill_gap(WEAK_STUDENT, branch="CSE", known_skills=[])
    tech_recs = [r for r in result["recommendations"] if r["Area"] == "Technical_Skills"]
    if tech_recs:
        assert "Consider learning" in tech_recs[0]["Advice"]


def test_subject_wise_breakdown_present_when_subscores_given():
    student = dict(WEAK_STUDENT)
    student.update({"DSA_Score": 20, "DBMS_Score": 25, "OS_CN_Score": 15,
                     "Quant_Score": 30, "Logical_Score": 35, "Verbal_Score": 28})
    result = analyze_skill_gap(student, branch="CSE")
    assert "technical_subject_gap" in result
    assert "aptitude_subject_gap" in result
    assert len(result["technical_subject_gap"]) == 3
    assert len(result["aptitude_subject_gap"]) == 3


def test_subject_wise_breakdown_absent_without_subscores():
    result = analyze_skill_gap(WEAK_STUDENT, branch="CSE")
    assert "technical_subject_gap" not in result
    assert "aptitude_subject_gap" not in result


def test_unknown_branch_falls_back_to_overall_profile():
    # Should not raise, and should still produce a full gap table.
    result = analyze_skill_gap(WEAK_STUDENT, branch="NOT_A_BRANCH")
    assert len(result["gap_table"]) == len(SKILL_FEATURES)
