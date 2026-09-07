"""
skills_catalog.py
------------------
A branch-wise catalog of specific, named skills (e.g. Python, Java, Power BI)
with rough importance weights, used to:
  1. Let students pick the actual skills they know instead of a single vague
     "Technical Skills" slider.
  2. Convert that selection into a Technical_Skills score (0-100) that feeds
     the existing trained ML model — so no retraining is required.
  3. Recommend the specific missing skills (not just "improve technical
     skills") in the Skill Gap report, ranked by importance.

NOTE: These weights are reasonable general-purpose defaults, not derived
from real placement data. Feel free to edit this file to match your
institution's curriculum / recruiter expectations — it's a plain Python
dict, no retraining needed after editing.
"""

# Skills common to (almost) every branch — general employability tools.
GENERAL_SKILLS = {
    "Power BI": 8,
    "Excel / Spreadsheets": 8,
    "Communication Tools (email, docs, presentations)": 5,
}

BRANCH_SKILLS = {
    "CSE": {
        "Python": 20,
        "Java": 15,
        "C++": 10,
        "SQL": 15,
        "Data Structures & Algorithms": 20,
        "Git / GitHub": 10,
        "Cloud (AWS / Azure / GCP)": 12,
        "Web Development (HTML/CSS/JS/React)": 12,
        "Machine Learning / AI": 12,
    },
    "IT": {
        "Python": 18,
        "Java": 12,
        "SQL": 18,
        "Data Structures & Algorithms": 15,
        "Git / GitHub": 10,
        "Cloud (AWS / Azure / GCP)": 15,
        "Web Development (HTML/CSS/JS/React)": 15,
        "Networking Basics": 10,
    },
    "ECE": {
        "C Programming": 15,
        "Python": 12,
        "MATLAB": 15,
        "Embedded Systems": 18,
        "VHDL / Verilog": 12,
        "IoT": 12,
        "PCB Design Tools": 10,
    },
    "EEE": {
        "MATLAB": 15,
        "Python": 10,
        "Power Systems Software (ETAP / PSCAD)": 18,
        "PLC Programming": 15,
        "AutoCAD Electrical": 15,
        "Embedded Systems": 12,
    },
    "MECH": {
        "AutoCAD": 18,
        "SolidWorks / CATIA": 18,
        "ANSYS": 15,
        "Python": 8,
        "MATLAB": 10,
        "CNC / Manufacturing Basics": 12,
    },
    "CIVIL": {
        "AutoCAD": 18,
        "STAAD Pro": 18,
        "Revit / BIM": 15,
        "Primavera / MS Project": 12,
        "Python": 6,
        "Surveying Tools": 12,
    },
}


# Illustrative learning-resource links shown next to each recommended skill
# in the "Skills to Learn Next" section. Free/well-known resources by
# default — feel free to swap in your institution's preferred platforms.
RESOURCE_LINKS = {
    "Python": "https://www.python.org/about/gettingstarted/",
    "Java": "https://dev.java/learn/",
    "C++": "https://cplusplus.com/doc/tutorial/",
    "C Programming": "https://www.learn-c.org/",
    "SQL": "https://www.w3schools.com/sql/",
    "Data Structures & Algorithms": "https://www.geeksforgeeks.org/data-structures/",
    "Git / GitHub": "https://docs.github.com/en/get-started",
    "Cloud (AWS / Azure / GCP)": "https://aws.amazon.com/getting-started/",
    "Web Development (HTML/CSS/JS/React)": "https://developer.mozilla.org/en-US/docs/Learn",
    "Machine Learning / AI": "https://www.coursera.org/learn/machine-learning",
    "Networking Basics": "https://www.cisco.com/c/en/us/solutions/enterprise-networks/what-is-networking.html",
    "MATLAB": "https://matlabacademy.mathworks.com/",
    "Embedded Systems": "https://www.embedded.com/learning-center/",
    "VHDL / Verilog": "https://www.chipverify.com/",
    "IoT": "https://www.arduino.cc/education/",
    "PCB Design Tools": "https://www.autodesk.com/products/eagle/free-download",
    "Power Systems Software (ETAP / PSCAD)": "https://etap.com/resources",
    "PLC Programming": "https://www.plcacademy.com/",
    "AutoCAD Electrical": "https://www.autodesk.com/learn/ondemand",
    "AutoCAD": "https://www.autodesk.com/learn/ondemand",
    "SolidWorks / CATIA": "https://www.solidworks.com/support/learning",
    "ANSYS": "https://innovationspace.ansys.com/courses/",
    "CNC / Manufacturing Basics": "https://www.tooling-u-sme.com/",
    "STAAD Pro": "https://www.bentley.com/software/staad/",
    "Revit / BIM": "https://www.autodesk.com/products/revit/learn",
    "Primavera / MS Project": "https://www.oracle.com/construction-engineering/primavera-p6/",
    "Surveying Tools": "https://www.nptel.ac.in/",
    "Power BI": "https://learn.microsoft.com/en-us/power-bi/fundamentals/",
    "Excel / Spreadsheets": "https://support.microsoft.com/en-us/excel",
    "Communication Tools (email, docs, presentations)": "https://edu.gcfglobal.org/en/topics/",
}


def get_resource_link(skill: str) -> str:
    """Return a learning-resource URL for a skill, or empty string if none."""
    return RESOURCE_LINKS.get(skill, "")


def get_skill_catalog(branch: str) -> dict:
    """Return {skill_name: weight} for a branch, including general skills."""
    branch_specific = BRANCH_SKILLS.get(branch, {})
    catalog = {**branch_specific, **GENERAL_SKILLS}
    return catalog


def compute_technical_score(branch: str, known_skills: list) -> float:
    """
    Convert a list of known skill names into a 0-100 Technical_Skills score,
    weighted by each skill's importance for the given branch.
    """
    catalog = get_skill_catalog(branch)
    total_weight = sum(catalog.values()) or 1
    earned_weight = sum(catalog.get(skill, 0) for skill in known_skills)
    score = (earned_weight / total_weight) * 100
    return round(min(score, 100), 1)


def get_missing_skills(branch: str, known_skills: list, top_n: int = 3) -> list:
    """
    Return the top_n highest-weighted skills the student has NOT selected,
    for the given branch — used to generate specific recommendations.
    """
    catalog = get_skill_catalog(branch)
    missing = {s: w for s, w in catalog.items() if s not in known_skills}
    ranked = sorted(missing.items(), key=lambda x: x[1], reverse=True)
    return [{"Skill": s, "Importance_Weight": w} for s, w in ranked[:top_n]]
