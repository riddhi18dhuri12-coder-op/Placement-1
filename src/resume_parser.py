"""
resume_parser.py
-----------------
Best-effort resume parser used to auto-fill the Predict tab from an
uploaded PDF/DOCX/TXT resume instead of manual entry.

This is intentionally heuristic (regex + keyword matching over plain
text) rather than a full NLP pipeline — resumes are too unstructured for
100% reliable extraction. Every field it returns is meant to pre-fill an
editable widget, not to be trusted blindly, so the app always shows the
student what was auto-detected and lets them correct it before predicting.

Extracted fields:
  - branch_guess            best-guess Branch code (CSE/IT/ECE/EEE/MECH/CIVIL)
  - cgpa                    float 0-10, or None if not confidently found
  - known_skills            list of catalog skill names found mentioned
  - projects_count          int, or None
  - internships_count       int, or None
  - certifications_count    int, or None
"""

import io
import re

# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text(uploaded_file) -> str:
    """
    uploaded_file: a Streamlit UploadedFile (has .name and behaves like a
    file-like object) or any file-like object with .read(). Returns plain
    text, or "" if the format isn't supported / extraction fails.
    """
    name = getattr(uploaded_file, "name", "") or ""
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    raw = uploaded_file.read()
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    try:
        if ext == "pdf":
            return _extract_pdf_text(raw)
        elif ext == "docx":
            return _extract_docx_text(raw)
        elif ext == "txt":
            return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""
    return ""


def _extract_pdf_text(raw: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(raw))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _extract_docx_text(raw: bytes) -> str:
    import docx
    document = docx.Document(io.BytesIO(raw))
    chunks = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                chunks.append(cell.text)
    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

_HEADER_NAMES = [
    "EDUCATION", "ACADEMIC BACKGROUND", "WORK EXPERIENCE", "EXPERIENCE",
    "INTERNSHIPS", "INTERNSHIP", "PROJECTS", "ACADEMIC PROJECTS",
    "PERSONAL PROJECTS", "CERTIFICATIONS", "CERTIFICATES", "SKILLS",
    "TECHNICAL SKILLS", "KEY SKILLS", "ACHIEVEMENTS", "AWARDS",
    "EXTRA CURRICULAR", "EXTRA-CURRICULAR ACTIVITIES", "PUBLICATIONS",
    "SUMMARY", "OBJECTIVE", "LANGUAGES", "HOBBIES", "DECLARATION",
    "CONTACT",
]
_HEADER_PATTERN = re.compile(
    r"(?im)^[ \t]*(" + "|".join(re.escape(h) for h in _HEADER_NAMES) + r")\s*:?\s*$"
)

_SECTION_ALIASES = {
    "projects": ["PROJECTS", "ACADEMIC PROJECTS", "PERSONAL PROJECTS"],
    "internships": ["INTERNSHIPS", "INTERNSHIP", "WORK EXPERIENCE", "EXPERIENCE"],
    "certifications": ["CERTIFICATIONS", "CERTIFICATES"],
    "education": ["EDUCATION", "ACADEMIC BACKGROUND"],
}


def _find_section(text: str, section_key: str) -> str:
    """Return the text between a matching header and the next header (or EOF)."""
    matches = list(_HEADER_PATTERN.finditer(text))
    wanted = set(_SECTION_ALIASES.get(section_key, []))
    for i, m in enumerate(matches):
        if m.group(1).upper() in wanted:
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            return text[start:end].strip()
    return ""


_BULLET_LINE = re.compile(r"(?m)^\s*[•▪●\-\*‣o]\s+\S")


def _count_entries(section_text: str) -> int:
    """Heuristic entry count within a resume section (bullets, else lines)."""
    if not section_text.strip():
        return 0
    bullets = _BULLET_LINE.findall(section_text)
    if bullets:
        return len(bullets)
    # No bullet markers: fall back to counting substantial, capitalized
    # lines (rough proxy for one entry per title/line).
    lines = [ln.strip() for ln in section_text.splitlines() if ln.strip()]
    substantial = [ln for ln in lines if len(ln) > 12]
    return max(1, len(substantial)) if substantial else 0


# ---------------------------------------------------------------------------
# CGPA extraction
# ---------------------------------------------------------------------------

_CGPA_OUT_OF_10 = re.compile(r"(?i)(?:cgpa|gpa)\s*[:\-]?\s*([0-9]\.\d{1,2}|10(?:\.0{1,2})?|[0-9])\s*/\s*10\b")
_CGPA_BARE = re.compile(r"(?i)\bcgpa\s*[:\-]?\s*([0-9]\.\d{1,2}|10(?:\.0{1,2})?|[0-9])\b")
_GPA_OUT_OF_4 = re.compile(r"(?i)\bgpa\s*[:\-]?\s*([0-4]\.\d{1,2}|[0-4])\s*/\s*4\b")
_PERCENT_AGG = re.compile(r"(?i)(?:aggregate|overall)\s*[:\-]?\s*([0-9]{1,3})\s*%")


def _extract_cgpa(text: str):
    m = _CGPA_OUT_OF_10.search(text)
    if m:
        val = float(m.group(1))
        return round(min(val, 10.0), 2)
    m = _GPA_OUT_OF_4.search(text)
    if m:
        val = float(m.group(1)) * 2.5
        return round(min(val, 10.0), 2)
    m = _CGPA_BARE.search(text)
    if m:
        val = float(m.group(1))
        if val <= 10.0:
            return round(val, 2)
    m = _PERCENT_AGG.search(text)
    if m:
        val = float(m.group(1))
        if val <= 100:
            return round(min(val / 10.0, 10.0), 2)
    return None


# ---------------------------------------------------------------------------
# Branch guessing
# ---------------------------------------------------------------------------

_BRANCH_KEYWORDS = {
    "CSE": ["computer science", "computer science and engineering", " cse ", "computer engineering"],
    "IT": ["information technology", " it engineering", "b.tech it", "b.tech. it"],
    "ECE": ["electronics and communication", "electronics & communication", " ece "],
    "EEE": ["electrical and electronics", "electrical & electronics", " eee "],
    "MECH": ["mechanical engineering", " mech "],
    "CIVIL": ["civil engineering", " civil "],
}


def _guess_branch(text: str):
    lower = f" {text.lower()} "
    for branch, keywords in _BRANCH_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return branch
    return None


# ---------------------------------------------------------------------------
# Skill matching
# ---------------------------------------------------------------------------

# Manually curated search terms for skills whose catalog name doesn't map
# cleanly to how people actually write it on a resume. Any catalog skill
# NOT listed here falls back to a generic term-extraction from its name.
SKILL_ALIASES = {
    "C Programming": ["c programming", "embedded c"],
    "C++": ["c++", "cpp"],
    "SQL": ["sql", "mysql", "postgresql", "pl/sql", "oracle db"],
    "Data Structures & Algorithms": ["data structures", "dsa", "algorithms"],
    "Git / GitHub": ["git", "github", "gitlab"],
    "Cloud (AWS / Azure / GCP)": ["aws", "amazon web services", "azure", "gcp", "google cloud"],
    "Web Development (HTML/CSS/JS/React)": ["html", "css", "javascript", "react", "web development", "reactjs"],
    "Machine Learning / AI": ["machine learning", "artificial intelligence", "deep learning"],
    "Networking Basics": ["networking", "computer networks", "ccna", "tcp/ip"],
    "Embedded Systems": ["embedded systems", "embedded"],
    "VHDL / Verilog": ["vhdl", "verilog"],
    "IoT": ["iot", "internet of things"],
    "PCB Design Tools": ["pcb design", "eagle cad", "kicad"],
    "Power Systems Software (ETAP / PSCAD)": ["etap", "pscad"],
    "PLC Programming": ["plc", "programmable logic controller"],
    "SolidWorks / CATIA": ["solidworks", "catia"],
    "CNC / Manufacturing Basics": ["cnc", "manufacturing"],
    "STAAD Pro": ["staad", "staad pro"],
    "Revit / BIM": ["revit", "bim"],
    "Primavera / MS Project": ["primavera", "ms project", "microsoft project"],
    "Surveying Tools": ["surveying", "total station"],
    "Power BI": ["power bi", "powerbi"],
    "Excel / Spreadsheets": ["excel", "spreadsheets", "google sheets"],
    "Communication Tools (email, docs, presentations)": ["ms office", "google docs"],
    "System Design": ["system design", "hld", "lld"],
    "REST API Design": ["rest api", "restful", "api design"],
    "JavaScript / TypeScript": ["javascript", "typescript"],
    "UI/UX Basics": ["ui/ux", "figma", "user experience"],
    "Testing (Jest / Cypress)": ["jest", "cypress"],
    "Node.js / Backend Frameworks": ["node.js", "nodejs", "express.js", "django", "flask", "spring boot"],
    "Databases (SQL/NoSQL)": ["mongodb", "nosql", "database design"],
    "Statistics & Probability": ["statistics", "probability"],
    "Data Visualization": ["data visualization", "tableau", "matplotlib", "seaborn"],
    "Pandas / NumPy": ["pandas", "numpy"],
    "Deep Learning (TensorFlow/PyTorch)": ["tensorflow", "pytorch", "keras", "deep learning"],
    "MLOps / Model Deployment": ["mlops", "model deployment", "mlflow"],
    "Linux / Shell Scripting": ["linux", "shell script", "bash scripting"],
    "CI/CD Tools (Jenkins/GitHub Actions)": ["jenkins", "ci/cd", "github actions", "gitlab ci"],
    "Docker / Kubernetes": ["docker", "kubernetes", "k8s"],
    "Security Fundamentals (OWASP, CIA Triad)": ["owasp", "cybersecurity", "network security"],
    "Ethical Hacking Tools": ["ethical hacking", "penetration testing", "kali linux", "burp suite"],
    "Cryptography Basics": ["cryptography", "encryption"],
    "Manual Testing Fundamentals": ["manual testing", "software testing"],
    "Test Automation (Selenium/Cypress)": ["selenium", "cypress", "test automation"],
    "API Testing (Postman)": ["postman", "api testing"],
    "Java / Python": ["java", "python"],
    "Bug Tracking Tools (Jira)": ["jira", "bug tracking"],
    "Requirement Gathering": ["requirement gathering", "brd", "frd"],
    "Business Domain Knowledge": ["business analysis", "domain knowledge"],
    "Microcontrollers (ARM/8051)": ["arm cortex", "8051", "microcontroller"],
    "RTOS Basics": ["rtos", "freertos"],
    "Digital Logic Design": ["digital logic", "logic design"],
    "ASIC/FPGA Design Flow": ["asic", "fpga"],
    "Synthesis Tools (Cadence/Synopsys)": ["cadence", "synopsys"],
    "Sensors & Communication Protocols": ["i2c", "spi protocol", "uart", "can protocol", "sensors"],
    "Electrical Machines Fundamentals": ["electrical machines"],
    "Power Systems Analysis": ["power systems"],
    "SCADA Systems": ["scada"],
    "Instrumentation Basics": ["instrumentation"],
    "GD&T Fundamentals": ["gd&t", "geometric dimensioning"],
    "Product Design Principles": ["product design"],
    "Lean Manufacturing / Six Sigma": ["lean manufacturing", "six sigma"],
    "Quality Control Tools": ["quality control"],
    "Production Planning": ["production planning"],
    "Structural Analysis Concepts": ["structural analysis"],
    "Construction Materials & Methods": ["construction materials"],
    "Quantity Estimation & Billing": ["quantity estimation", "boq"],
    "Site Safety Management": ["site safety"],
}


def _search_terms_for(skill: str) -> list:
    if skill in SKILL_ALIASES:
        return SKILL_ALIASES[skill]
    # Generic fallback: strip a parenthetical, split on separators.
    main = re.sub(r"\([^)]*\)", "", skill).strip()
    terms = set()
    if main:
        terms.add(main.lower())
    for part in re.split(r"[\/,&]", main):
        part = part.strip().lower()
        if len(part) > 1:
            terms.add(part)
    return list(terms)


def find_known_skills(text: str, skill_catalog: dict) -> list:
    """
    skill_catalog: {skill_name: weight} — typically the combined
    branch + role catalog so anything relevant can be detected.
    Returns the subset of skill_catalog keys mentioned in `text`.
    """
    lower = text.lower()
    found = []
    for skill in skill_catalog:
        for term in _search_terms_for(skill):
            term = term.strip().lower()
            if not term:
                continue
            pattern = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
            if re.search(pattern, lower):
                found.append(skill)
                break
    return found


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_resume_text(text: str, skill_catalog: dict) -> dict:
    """Parse already-extracted resume text into form-fillable fields."""
    projects_section = _find_section(text, "projects")
    internships_section = _find_section(text, "internships")
    certifications_section = _find_section(text, "certifications")

    projects_count = _count_entries(projects_section) if projects_section else None
    internships_count = _count_entries(internships_section) if internships_section else None
    certifications_count = _count_entries(certifications_section) if certifications_section else None

    # Cap to the same ranges the UI's number_input widgets allow.
    if projects_count is not None:
        projects_count = min(projects_count, 15)
    if internships_count is not None:
        internships_count = min(internships_count, 10)
    if certifications_count is not None:
        certifications_count = min(certifications_count, 15)

    return {
        "branch_guess": _guess_branch(text),
        "cgpa": _extract_cgpa(text),
        "known_skills": find_known_skills(text, skill_catalog),
        "projects_count": projects_count,
        "internships_count": internships_count,
        "certifications_count": certifications_count,
        "text_extracted": bool(text.strip()),
    }


def parse_resume_file(uploaded_file, skill_catalog: dict) -> dict:
    """Extract text from an uploaded resume file, then parse it."""
    text = extract_text(uploaded_file)
    result = parse_resume_text(text, skill_catalog)
    result["raw_text"] = text
    return result
