"""
job_roles.py
------------
Job-role-based skill catalog, used to power role-specific skill gap
recommendations: the student picks the job role they're targeting (e.g.
"Data Scientist", "Embedded Systems Engineer") in addition to their branch,
and recommendations are ranked against what THAT ROLE actually needs,
instead of only a generic branch-wide skill list.

Design mirrors skills_catalog.py on purpose (same {skill: weight} shape)
so the two catalogs can be merged and reused by the same UI/report code.

NOTE: Weights are reasonable general-purpose defaults reflecting typical
recruiter/job-description emphasis, not derived from real hiring data.
Edit freely — plain Python dicts, no retraining needed.
"""

from skills_catalog import RESOURCE_LINKS as _BASE_RESOURCE_LINKS

JOB_ROLES = {
    # ---------------- CSE / IT ----------------
    "Software Development Engineer (Backend)": {
        "Data Structures & Algorithms": 20,
        "Python": 12,
        "Java": 12,
        "SQL": 15,
        "System Design": 15,
        "Git / GitHub": 8,
        "REST API Design": 10,
        "Cloud (AWS / Azure / GCP)": 8,
    },
    "Frontend Developer": {
        "Web Development (HTML/CSS/JS/React)": 25,
        "JavaScript / TypeScript": 20,
        "UI/UX Basics": 12,
        "Git / GitHub": 10,
        "REST API Design": 10,
        "Testing (Jest / Cypress)": 8,
        "Performance Optimization": 7,
        "Responsive Design": 8,
    },
    "Full Stack Developer": {
        "Web Development (HTML/CSS/JS/React)": 18,
        "Node.js / Backend Frameworks": 15,
        "SQL": 12,
        "Databases (SQL/NoSQL)": 10,
        "Git / GitHub": 8,
        "REST API Design": 12,
        "Cloud (AWS / Azure / GCP)": 10,
        "Data Structures & Algorithms": 15,
    },
    "Data Analyst": {
        "SQL": 22,
        "Excel / Spreadsheets": 15,
        "Power BI": 15,
        "Python": 12,
        "Statistics": 15,
        "Data Visualization": 12,
        "Communication Tools (email, docs, presentations)": 9,
    },
    "Data Scientist": {
        "Python": 18,
        "Statistics & Probability": 16,
        "Machine Learning / AI": 18,
        "SQL": 12,
        "Pandas / NumPy": 12,
        "Data Visualization": 10,
        "Power BI": 6,
        "Communication Tools (email, docs, presentations)": 8,
    },
    "Machine Learning Engineer": {
        "Python": 16,
        "Machine Learning / AI": 20,
        "Deep Learning (TensorFlow/PyTorch)": 18,
        "SQL": 8,
        "Statistics & Probability": 10,
        "MLOps / Model Deployment": 14,
        "Cloud (AWS / Azure / GCP)": 8,
        "Data Structures & Algorithms": 6,
    },
    "DevOps / Cloud Engineer": {
        "Cloud (AWS / Azure / GCP)": 22,
        "Linux / Shell Scripting": 16,
        "CI/CD Tools (Jenkins/GitHub Actions)": 16,
        "Docker / Kubernetes": 18,
        "Git / GitHub": 8,
        "Networking Basics": 10,
        "Python": 10,
    },
    "Cybersecurity Analyst": {
        "Networking Basics": 16,
        "Security Fundamentals (OWASP, CIA Triad)": 18,
        "Linux / Shell Scripting": 12,
        "Ethical Hacking Tools": 16,
        "Cloud (AWS / Azure / GCP)": 10,
        "Python": 10,
        "Cryptography Basics": 10,
        "SQL": 8,
    },
    "QA / Test Automation Engineer": {
        "Manual Testing Fundamentals": 16,
        "Test Automation (Selenium/Cypress)": 20,
        "SQL": 10,
        "API Testing (Postman)": 14,
        "Java / Python": 14,
        "Git / GitHub": 8,
        "Bug Tracking Tools (Jira)": 10,
        "Communication Tools (email, docs, presentations)": 8,
    },
    "Business Analyst": {
        "Excel / Spreadsheets": 18,
        "SQL": 16,
        "Power BI": 16,
        "Communication Tools (email, docs, presentations)": 14,
        "Requirement Gathering": 14,
        "Business Domain Knowledge": 12,
        "Data Visualization": 10,
    },
    # ---------------- ECE ----------------
    "Embedded Systems Engineer": {
        "C Programming": 20,
        "Embedded Systems": 20,
        "Microcontrollers (ARM/8051)": 16,
        "RTOS Basics": 12,
        "VHDL / Verilog": 8,
        "IoT": 10,
        "PCB Design Tools": 8,
        "Python": 6,
    },
    "VLSI / Chip Design Engineer": {
        "VHDL / Verilog": 24,
        "Digital Logic Design": 18,
        "ASIC/FPGA Design Flow": 18,
        "Synthesis Tools (Cadence/Synopsys)": 16,
        "C Programming": 8,
        "Embedded Systems": 8,
        "MATLAB": 8,
    },
    "IoT Engineer": {
        "IoT": 20,
        "Embedded Systems": 16,
        "C Programming": 12,
        "Python": 12,
        "Networking Basics": 12,
        "Cloud (AWS / Azure / GCP)": 14,
        "PCB Design Tools": 8,
        "Sensors & Communication Protocols": 6,
    },
    # ---------------- EEE ----------------
    "Electrical Design Engineer": {
        "AutoCAD Electrical": 20,
        "Power Systems Software (ETAP / PSCAD)": 20,
        "MATLAB": 14,
        "Electrical Machines Fundamentals": 14,
        "Power Systems Analysis": 14,
        "Embedded Systems": 8,
        "Python": 6,
    },
    "Automation & Controls Engineer (PLC/SCADA)": {
        "PLC Programming": 24,
        "SCADA Systems": 18,
        "Embedded Systems": 12,
        "AutoCAD Electrical": 12,
        "Instrumentation Basics": 14,
        "MATLAB": 10,
        "Python": 6,
        "Networking Basics": 4,
    },
    # ---------------- MECH ----------------
    "Mechanical Design Engineer": {
        "SolidWorks / CATIA": 22,
        "AutoCAD": 18,
        "GD&T Fundamentals": 14,
        "ANSYS": 16,
        "Product Design Principles": 12,
        "MATLAB": 8,
        "CNC / Manufacturing Basics": 10,
    },
    "Manufacturing / Production Engineer": {
        "CNC / Manufacturing Basics": 22,
        "Lean Manufacturing / Six Sigma": 18,
        "AutoCAD": 14,
        "Quality Control Tools": 14,
        "SolidWorks / CATIA": 12,
        "Production Planning": 12,
        "Excel / Spreadsheets": 8,
    },
    # ---------------- CIVIL ----------------
    "Structural Design Engineer": {
        "STAAD Pro": 22,
        "AutoCAD": 18,
        "Revit / BIM": 16,
        "Structural Analysis Concepts": 16,
        "ETABS": 14,
        "Surveying Tools": 6,
        "Excel / Spreadsheets": 8,
    },
    "Site / Construction Engineer": {
        "Primavera / MS Project": 20,
        "Surveying Tools": 18,
        "AutoCAD": 14,
        "Construction Materials & Methods": 16,
        "Quantity Estimation & Billing": 16,
        "Site Safety Management": 10,
        "Communication Tools (email, docs, presentations)": 6,
    },
}

# Roles typically pursued by students from each branch — used only to put
# the most relevant roles first in the dropdown. Students can still pick
# ANY role regardless of branch (e.g. a MECH student targeting Data Analyst).
ROLES_BY_BRANCH = {
    "CSE": [
        "Software Development Engineer (Backend)", "Full Stack Developer",
        "Frontend Developer", "Data Scientist", "Machine Learning Engineer",
        "DevOps / Cloud Engineer", "Cybersecurity Analyst",
        "QA / Test Automation Engineer", "Data Analyst", "Business Analyst",
    ],
    "IT": [
        "Full Stack Developer", "DevOps / Cloud Engineer", "Data Analyst",
        "Software Development Engineer (Backend)", "Cybersecurity Analyst",
        "QA / Test Automation Engineer", "Frontend Developer", "Business Analyst",
    ],
    "ECE": [
        "Embedded Systems Engineer", "VLSI / Chip Design Engineer", "IoT Engineer",
        "Software Development Engineer (Backend)", "Data Analyst",
    ],
    "EEE": [
        "Electrical Design Engineer", "Automation & Controls Engineer (PLC/SCADA)",
        "Embedded Systems Engineer", "Data Analyst",
    ],
    "MECH": [
        "Mechanical Design Engineer", "Manufacturing / Production Engineer",
        "Data Analyst", "Business Analyst",
    ],
    "CIVIL": [
        "Structural Design Engineer", "Site / Construction Engineer", "Business Analyst",
    ],
}

# Extra resource links for role-specific skills not already covered in
# skills_catalog.RESOURCE_LINKS.
ROLE_RESOURCE_LINKS = {
    "System Design": "https://github.com/donnemartin/system-design-primer",
    "REST API Design": "https://restfulapi.net/",
    "JavaScript / TypeScript": "https://www.typescriptlang.org/docs/",
    "UI/UX Basics": "https://www.interaction-design.org/",
    "Testing (Jest / Cypress)": "https://docs.cypress.io/guides/overview/why-cypress",
    "Performance Optimization": "https://web.dev/learn/performance/",
    "Responsive Design": "https://developer.mozilla.org/en-US/docs/Learn/CSS/CSS_layout/Responsive_Design",
    "Node.js / Backend Frameworks": "https://nodejs.org/en/learn",
    "Databases (SQL/NoSQL)": "https://www.mongodb.com/nosql-explained",
    "Statistics": "https://www.khanacademy.org/math/statistics-probability",
    "Statistics & Probability": "https://www.khanacademy.org/math/statistics-probability",
    "Data Visualization": "https://www.tableau.com/learn/training",
    "Pandas / NumPy": "https://pandas.pydata.org/docs/getting_started/index.html",
    "Deep Learning (TensorFlow/PyTorch)": "https://pytorch.org/tutorials/",
    "MLOps / Model Deployment": "https://ml-ops.org/",
    "Linux / Shell Scripting": "https://linuxjourney.com/",
    "CI/CD Tools (Jenkins/GitHub Actions)": "https://docs.github.com/en/actions",
    "Docker / Kubernetes": "https://kubernetes.io/docs/tutorials/",
    "Security Fundamentals (OWASP, CIA Triad)": "https://owasp.org/www-project-top-ten/",
    "Ethical Hacking Tools": "https://www.offsec.com/courses/pen-200/",
    "Cryptography Basics": "https://cryptography101.org/",
    "Manual Testing Fundamentals": "https://www.guru99.com/software-testing.html",
    "Test Automation (Selenium/Cypress)": "https://www.selenium.dev/documentation/",
    "API Testing (Postman)": "https://learning.postman.com/",
    "Java / Python": "https://dev.java/learn/",
    "Bug Tracking Tools (Jira)": "https://www.atlassian.com/software/jira/guides",
    "Requirement Gathering": "https://www.iiba.org/career-resources/",
    "Business Domain Knowledge": "https://www.coursera.org/browse/business",
    "Microcontrollers (ARM/8051)": "https://www.arm.com/resources/education",
    "RTOS Basics": "https://www.freertos.org/",
    "Digital Logic Design": "https://www.nptel.ac.in/",
    "ASIC/FPGA Design Flow": "https://www.chipverify.com/",
    "Synthesis Tools (Cadence/Synopsys)": "https://www.synopsys.com/glossary.html",
    "Sensors & Communication Protocols": "https://www.arduino.cc/education/",
    "Electrical Machines Fundamentals": "https://www.nptel.ac.in/",
    "Power Systems Analysis": "https://www.nptel.ac.in/",
    "SCADA Systems": "https://www.plcacademy.com/scada-tutorial/",
    "Instrumentation Basics": "https://www.nptel.ac.in/",
    "GD&T Fundamentals": "https://www.asme.org/learning-development",
    "Product Design Principles": "https://www.coursera.org/browse/arts-and-humanities/design",
    "Lean Manufacturing / Six Sigma": "https://asq.org/quality-resources/six-sigma",
    "Quality Control Tools": "https://asq.org/quality-resources/",
    "Production Planning": "https://www.nptel.ac.in/",
    "Structural Analysis Concepts": "https://www.nptel.ac.in/",
    "ETABS": "https://www.csiamerica.com/products/etabs/watch-and-learn",
    "Construction Materials & Methods": "https://www.nptel.ac.in/",
    "Quantity Estimation & Billing": "https://www.nptel.ac.in/",
    "Site Safety Management": "https://www.nsc.org/workplace/safety-topics",
}


def list_roles() -> list:
    """All available job roles, alphabetically sorted."""
    return sorted(JOB_ROLES.keys())


def ordered_roles_for_branch(branch: str) -> list:
    """
    Roles most relevant to `branch` first, followed by every other role —
    so the dropdown defaults sensibly but never hides cross-branch options.
    """
    preferred = ROLES_BY_BRANCH.get(branch, [])
    rest = [r for r in list_roles() if r not in preferred]
    return preferred + rest


def get_role_catalog(role: str) -> dict:
    """Return {skill_name: weight} required for a given job role."""
    return JOB_ROLES.get(role, {})


def get_master_skill_catalog() -> dict:
    """
    Union of every branch's skill catalog and every role's skill catalog —
    used when we don't yet know the student's branch/role (e.g. scanning a
    freshly uploaded resume for skill mentions before the form is filled in).
    """
    from skills_catalog import BRANCH_SKILLS, GENERAL_SKILLS
    combined = dict(GENERAL_SKILLS)
    for branch_catalog in BRANCH_SKILLS.values():
        combined.update(branch_catalog)
    for role_catalog in JOB_ROLES.values():
        combined.update(role_catalog)
    return combined


def guess_best_role(known_skills: list) -> str:
    """
    Best-matching role for a set of known skills, by total weight overlap.
    Returns None if no role has any overlap (e.g. no skills detected).
    """
    if not known_skills:
        return None
    known = set(known_skills)
    best_role, best_score = None, 0
    for role, catalog in JOB_ROLES.items():
        score = sum(w for s, w in catalog.items() if s in known)
        if score > best_score:
            best_role, best_score = role, score
    return best_role


def get_combined_catalog(branch: str, role: str) -> dict:
    """
    Union of the branch-wide skill catalog and the role-specific catalog,
    for populating the "select what you know" multiselect — so a student
    can select skills their role needs even if it's outside their branch's
    default list. Role weights take precedence on overlap since the role
    is what the student says they're targeting.
    """
    from skills_catalog import get_skill_catalog
    combined = dict(get_skill_catalog(branch))
    combined.update(get_role_catalog(role))
    return combined


def compute_role_readiness(role: str, known_skills: list) -> float:
    """
    0-100 score: how much of the target role's weighted skill requirement
    the student's selected skills already cover.
    """
    catalog = get_role_catalog(role)
    if not catalog:
        return 0.0
    total_weight = sum(catalog.values()) or 1
    earned_weight = sum(w for s, w in catalog.items() if s in (known_skills or []))
    return round(min(earned_weight / total_weight * 100, 100), 1)


def get_missing_role_skills(role: str, known_skills: list, top_n: int = 5) -> list:
    """Top_n highest-weighted skills required for `role` the student lacks."""
    catalog = get_role_catalog(role)
    known = known_skills or []
    missing = {s: w for s, w in catalog.items() if s not in known}
    ranked = sorted(missing.items(), key=lambda x: x[1], reverse=True)
    return [{"Skill": s, "Importance_Weight": w} for s, w in ranked[:top_n]]


def get_role_resource_link(skill: str) -> str:
    """Learning-resource URL for a skill, checking role links then the base catalog."""
    if skill in ROLE_RESOURCE_LINKS:
        return ROLE_RESOURCE_LINKS[skill]
    return _BASE_RESOURCE_LINKS.get(skill, "")
