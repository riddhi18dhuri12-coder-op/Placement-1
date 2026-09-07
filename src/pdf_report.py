"""
pdf_report.py
--------------
Builds a downloadable PDF summary of a student's placement prediction and
skill-gap analysis, so they can save/print/share it outside the dashboard.
"""

import datetime
import os
import tempfile
from fpdf import FPDF

# The built-in "Helvetica" core font only supports latin-1, so any text
# rendered with it (em-dashes, curly quotes, etc. from the app's copy)
# needs to be transliterated to plain ASCII/latin-1 equivalents first.
_CHAR_MAP = {
    "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2192": "->",
    "\u2705": "[OK]", "\u274c": "[X]",
}


def _sanitize(text) -> str:
    text = str(text)
    for src, dst in _CHAR_MAP.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(60, 40, 140)
        self.cell(0, 10, "Placement Prediction & Skill Gap Report", ln=True, align="C")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, datetime.datetime.now().strftime("Generated %d %b %Y, %H:%M"),
                   ln=True, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, text):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 30, 30)
        self.ln(3)
        self.cell(0, 8, _sanitize(text), ln=True)
        self.set_draw_color(108, 92, 231)
        self.set_line_width(0.6)
        self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
        self.ln(3)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10.5)
        self.set_text_color(20, 20, 20)
        self.set_x(self.l_margin)
        self.multi_cell(0, 6, _sanitize(text))
        self.set_x(self.l_margin)

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        self.set_font("Helvetica", "B", 9.5)
        self.set_fill_color(230, 226, 250)
        for h, w in zip(headers, col_widths):
            self.cell(w, 7, _sanitize(h), border=1, fill=True)
        self.ln()
        self.set_font("Helvetica", "", 9.5)
        self.set_fill_color(250, 250, 252)
        fill = False
        for row in rows:
            for val, w in zip(row, col_widths):
                self.cell(w, 6.5, _sanitize(val), border=1, fill=fill)
            self.ln()
            fill = not fill


def build_report_pdf(
    student_id: str,
    branch: str,
    prediction_label: str,
    probability: float,
    inputs: dict,
    gap_rows: list,
    recommendations: list,
    missing_skills: list = None,
    eligible_companies: list = None,
    output_path: str = "/tmp/placement_report.pdf",
) -> str:
    if output_path is None:
            output_path = os.path.join(tempfile.gettempdir(),"placement_report.pdf")
    pdf = ReportPDF()
    pdf.add_page()

    pdf.section_title("Student Summary")
    pdf.body_text(
        f"Student ID: {student_id or 'N/A'}\n"
        f"Branch: {branch}\n"
        f"Prediction: {prediction_label}\n"
        f"Estimated placement probability: {probability * 100:.1f}%"
    )

    pdf.section_title("Inputs")
    pdf.table(
        headers=["Field", "Value"],
        rows=[[k.replace("_", " "), v] for k, v in inputs.items()],
        col_widths=[95, 95],
    )

    if gap_rows:
        pdf.section_title("Skill Gap vs. Placed Peers (same branch)")
        pdf.table(
            headers=["Feature", "Your Value", "Benchmark", "Gap %"],
            rows=[[r["Feature"].replace("_", " "), r["Your_Value"],
                   r["Benchmark_Value"], f"{r['Gap_Percent']}%"] for r in gap_rows],
            col_widths=[70, 40, 40, 40],
        )

    if recommendations:
        pdf.section_title("Personalized Recommendations")
        for rec in recommendations:
            pdf.body_text(f"- {rec['Area'].replace('_', ' ')}: {rec['Advice']}")
            pdf.ln(1)

    if missing_skills:
        pdf.section_title("Skills to Learn Next")
        pdf.body_text(", ".join(m["Skill"] for m in missing_skills))

    if eligible_companies:
        pdf.section_title("Illustrative Company Eligibility")
        pdf.body_text(
            "Based on editable example criteria (not live recruiter data) — "
            "see config/company_criteria.json:"
        )
        for c in eligible_companies:
            mark = "ELIGIBLE" if c["eligible"] else "NOT YET"
            detail = "" if c["eligible"] else " (" + "; ".join(c["reasons"]) + ")"
            pdf.body_text(f"- {c['company']}: {mark}{detail}")

    pdf.section_title("Disclaimer")
    pdf.body_text(
        "This report is generated from a trained machine-learning model using "
        "historical/simulated data. It is meant to guide preparation and is not "
        "a guarantee of any placement outcome."
    )

    pdf.output(output_path)
    return output_path
