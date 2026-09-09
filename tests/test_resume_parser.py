import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from resume_parser import parse_resume_text, find_known_skills, _extract_cgpa, _guess_branch  # noqa: E402
from job_roles import get_combined_catalog  # noqa: E402

SAMPLE_RESUME = """
John Doe
B.Tech, Computer Science and Engineering, XYZ University
CGPA: 8.4/10

TECHNICAL SKILLS
Python, Java, SQL, Git, React, AWS, Machine Learning

PROJECTS
- Built a food delivery web app using React and Node.js
- Created a machine learning model to predict house prices using Python and pandas
- Developed a REST API for a library management system

INTERNSHIPS
- Software Engineering Intern at Acme Corp, Summer 2025
- Data Analyst Intern at Beta Inc, Winter 2024

CERTIFICATIONS
- AWS Certified Cloud Practitioner
- Google Data Analytics Certificate
"""


def test_cgpa_extraction_out_of_10():
    assert _extract_cgpa("CGPA: 8.4/10") == 8.4


def test_cgpa_extraction_gpa_out_of_4():
    assert _extract_cgpa("GPA: 3.6/4") == 9.0


def test_cgpa_extraction_percentage_fallback():
    assert _extract_cgpa("Aggregate: 82%") == 8.2


def test_cgpa_extraction_returns_none_when_absent():
    assert _extract_cgpa("No academic score mentioned here.") is None


def test_branch_guess_from_degree_line():
    assert _guess_branch("B.Tech, Mechanical Engineering") == "MECH"
    assert _guess_branch("B.Tech, Civil Engineering") == "CIVIL"
    assert _guess_branch("Completely unrelated text") is None


def test_find_known_skills_matches_catalog_terms():
    catalog = get_combined_catalog("CSE", "Data Scientist")
    found = find_known_skills("Experienced with Python, AWS and React.", catalog)
    assert "Python" in found
    assert "Cloud (AWS / Azure / GCP)" in found
    assert "Web Development (HTML/CSS/JS/React)" in found


def test_parse_resume_text_full_pipeline():
    catalog = get_combined_catalog("CSE", "Data Scientist")
    result = parse_resume_text(SAMPLE_RESUME, catalog)
    assert result["branch_guess"] == "CSE"
    assert result["cgpa"] == 8.4
    assert result["projects_count"] == 3
    assert result["internships_count"] == 2
    assert result["certifications_count"] == 2
    assert "Python" in result["known_skills"]
    assert "Pandas / NumPy" in result["known_skills"]


def test_parse_resume_text_handles_empty_text():
    catalog = get_combined_catalog("CSE", "Data Scientist")
    result = parse_resume_text("", catalog)
    assert result["cgpa"] is None
    assert result["known_skills"] == []
    assert result["projects_count"] is None
    assert result["text_extracted"] is False


def test_counts_are_capped_to_ui_ranges():
    many_projects = "PROJECTS\n" + "\n".join(f"- Project number {i} description here" for i in range(30))
    catalog = get_combined_catalog("CSE", "Data Scientist")
    result = parse_resume_text(many_projects, catalog)
    assert result["projects_count"] <= 15
