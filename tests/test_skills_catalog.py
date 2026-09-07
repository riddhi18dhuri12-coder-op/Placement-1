import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from skills_catalog import (  # noqa: E402
    get_skill_catalog, compute_technical_score, get_missing_skills, get_resource_link,
)


def test_catalog_includes_general_skills_for_every_branch():
    for branch in ["CSE", "IT", "ECE", "EEE", "MECH", "CIVIL"]:
        catalog = get_skill_catalog(branch)
        assert "Power BI" in catalog
        assert len(catalog) > len(get_skill_catalog("__unknown__"))


def test_unknown_branch_returns_only_general_skills():
    catalog = get_skill_catalog("NOT_A_BRANCH")
    assert set(catalog.keys()) == {"Power BI", "Excel / Spreadsheets",
                                    "Communication Tools (email, docs, presentations)"}


def test_technical_score_zero_when_no_skills_known():
    assert compute_technical_score("CSE", []) == 0


def test_technical_score_full_when_all_skills_known():
    catalog = get_skill_catalog("CSE")
    score = compute_technical_score("CSE", list(catalog.keys()))
    assert score == 100.0


def test_technical_score_is_between_0_and_100():
    score = compute_technical_score("CSE", ["Python", "SQL"])
    assert 0 <= score <= 100


def test_missing_skills_excludes_known_skills():
    known = ["Python", "SQL"]
    missing = get_missing_skills("CSE", known, top_n=10)
    missing_names = [m["Skill"] for m in missing]
    assert "Python" not in missing_names
    assert "SQL" not in missing_names


def test_missing_skills_ranked_by_importance_descending():
    missing = get_missing_skills("CSE", [], top_n=5)
    weights = [m["Importance_Weight"] for m in missing]
    assert weights == sorted(weights, reverse=True)


def test_missing_skills_respects_top_n():
    missing = get_missing_skills("CSE", [], top_n=3)
    assert len(missing) == 3


def test_resource_link_returns_url_for_known_skill():
    assert get_resource_link("Python").startswith("http")


def test_resource_link_empty_for_unknown_skill():
    assert get_resource_link("Not A Real Skill") == ""
