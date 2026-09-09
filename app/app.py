"""
app.py
------
Streamlit dashboard for the Placement Prediction & Skill Gap Analysis system.

Run with:
    streamlit run app/app.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from skill_gap import analyze_skill_gap, SKILL_FEATURES  # noqa: E402
from skills_catalog import get_skill_catalog, compute_technical_score, get_resource_link  # noqa: E402
from job_roles import (  # noqa: E402
    ordered_roles_for_branch, get_combined_catalog, get_role_resource_link,
    get_master_skill_catalog, guess_best_role,
)
from resume_parser import parse_resume_file  # noqa: E402
from company_eligibility import check_eligibility  # noqa: E402
from progress_store import save_snapshot, load_history, clear_history  # noqa: E402
from pdf_report import build_report_pdf  # noqa: E402

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

FEATURE_ORDER = [
    "CGPA", "Aptitude_Score", "Technical_Skills", "Communication_Skills",
    "Projects_Count", "Internships_Count", "Certifications_Count",
    "Backlogs", "Branch"
]

st.set_page_config(
    page_title="Placement Predictor & Skill Gap Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Custom CSS — card styling, hero header, accent colors
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .hero {
        background: linear-gradient(135deg, #6C5CE7 0%, #341F97 100%);
        padding: 2rem 2.2rem;
        border-radius: 16px;
        margin-bottom: 1.6rem;
    }
    .hero h1 { color: white; font-size: 2.1rem; margin-bottom: 0.3rem; }
    .hero p { color: #E4E1FF; font-size: 1.02rem; margin: 0; }
    .metric-card {
        background: #1A1D29; border: 1px solid #2E3140; border-radius: 14px;
        padding: 1.1rem 1.3rem; text-align: center;
    }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #A29BFE; }
    .metric-card .label { font-size: 0.85rem; color: #9A9CB0; margin-top: 0.2rem; }
    .skill-pill {
        display: inline-block; background: #2E2A5C; color: #D8D3FF;
        border-radius: 999px; padding: 0.25rem 0.8rem; margin: 0.15rem; font-size: 0.85rem;
    }
    .skill-pill a { color: #D8D3FF; text-decoration: none; }
    .rec-card {
        background: #1A1D29; border-left: 4px solid #6C5CE7; border-radius: 10px;
        padding: 0.9rem 1.1rem; margin-bottom: 0.7rem;
    }
    .rec-card b { color: #A29BFE; }
    .company-card {
        background: #1A1D29; border-radius: 10px; padding: 0.7rem 1rem;
        margin-bottom: 0.5rem; border-left: 4px solid #2E3140;
    }
    .company-eligible { border-left-color: #00D2A0 !important; }
    .company-not-eligible { border-left-color: #FF6B6B !important; }
    section[data-testid="stSidebar"] { background-color: #14161F; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🎓 Placement Predictor & Skill Gap Analyzer</h1>
    <p>Estimate your placement probability and get a personalized, skill-by-skill improvement plan.</p>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    branch_encoder = joblib.load(os.path.join(MODELS_DIR, "branch_encoder.pkl"))
    raw_model_path = os.path.join(MODELS_DIR, "best_model_raw.pkl")
    raw_model = joblib.load(raw_model_path) if os.path.exists(raw_model_path) else None
    unc_path = os.path.join(MODELS_DIR, "uncertainty_model.pkl")
    uncertainty_model = joblib.load(unc_path) if os.path.exists(unc_path) else None
    return model, scaler, branch_encoder, raw_model, uncertainty_model


@st.cache_data
def load_dataset_stats():
    df = pd.read_csv(os.path.join(DATA_DIR, "placement_data.csv"))
    return df


@st.cache_data
def load_model_info():
    path = os.path.join(MODELS_DIR, "best_model_info.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_resource
def get_shap_explainer(_raw_model, _background):
    import shap
    if hasattr(_raw_model, "feature_importances_"):
        return shap.TreeExplainer(_raw_model)
    try:
        return shap.LinearExplainer(_raw_model, _background)
    except Exception:
        return shap.Explainer(_raw_model.predict_proba, _background)


def build_feature_row(student_input: dict, branch: str, branch_encoder) -> pd.DataFrame:
    row = student_input.copy()
    row["Branch"] = branch_encoder.transform([branch])[0]
    return pd.DataFrame([row])[FEATURE_ORDER]


def predict_with_uncertainty(model, uncertainty_model, scaler, X_scaled):
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1]
    band = None
    if uncertainty_model is not None:
        tree_probs = np.array([t.predict_proba(X_scaled)[0][1] for t in uncertainty_model.estimators_])
        band = float(tree_probs.std())
    return pred, proba, band


artifacts_ready = True
try:
    model, scaler, branch_encoder, raw_model, uncertainty_model = load_artifacts()
except FileNotFoundError:
    artifacts_ready = False

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# ----------------------------------------------------------------------------
# Sidebar — project snapshot + student ID for progress tracking
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📌 Project Snapshot")
    if not artifacts_ready:
        st.warning(
            "Model artifacts not found. Run:\n\n"
            "`python src/generate_dataset.py`\n\n"
            "`python src/eda.py`\n\n"
            "`python src/train_models.py`"
        )
    else:
        df_full = load_dataset_stats()
        info = load_model_info()
        st.metric("Students in Dataset", f"{len(df_full):,}")
        st.metric("Historical Placement Rate", f"{df_full['Placement_Status'].mean()*100:.1f}%")
        if info:
            st.metric("Best Model", info["best_model"])
            st.metric("Model F1-Score", f"{info['metrics']['F1_Score']*100:.1f}%")
    st.markdown("---")
    st.markdown("### 👤 Student ID (for progress tracking)")
    student_id = st.text_input(
        "Enter an ID (roll no. / name)", value=st.session_state.get("student_id", ""),
        help="Used only to separate your saved history from others on this device. "
             "Not a real login — no password, just a label.",
    )
    st.session_state["student_id"] = student_id
    st.markdown("---")
    st.caption(
        "Built with scikit-learn, XGBoost, SHAP & Streamlit. "
        "Predictions are guidance based on trained ML models, not guarantees."
    )

tab_predict, tab_whatif, tab_batch, tab_progress, tab_insights, tab_about = st.tabs(
    ["🎯 Predict", "🔧 What-If", "📁 Batch", "📈 My Progress", "📊 Model Insights", "ℹ️ About"]
)

# ============================================================================
# TAB 1 — PREDICT
# ============================================================================
with tab_predict:
    st.subheader("Your Details")

    # ---- Defaults (only take effect the first time each key is touched) ----
    st.session_state.setdefault("branch_select", "CSE")
    st.session_state.setdefault("cgpa_input", 7.0)
    st.session_state.setdefault("aptitude_input", 60)
    st.session_state.setdefault("communication_input", 60)
    st.session_state.setdefault("projects_input", 2)
    st.session_state.setdefault("internships_input", 0)
    st.session_state.setdefault("certifications_input", 1)
    st.session_state.setdefault("backlogs_input", 0)
    st.session_state.setdefault("known_skills_select", [])

    # ---- Resume upload (alternative to manual entry) ----
    st.markdown("**📄 Upload your resume to auto-fill this form** *(optional — you can still edit everything below)*")
    resume_file = st.file_uploader(
        "Resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"], label_visibility="collapsed",
    )
    resume_applied_key = f"{resume_file.name}_{resume_file.size}" if resume_file is not None else None

    if resume_file is not None and st.session_state.get("_resume_applied_key") != resume_applied_key:
        with st.spinner("Reading your resume..."):
            parsed = parse_resume_file(resume_file, get_master_skill_catalog())
        st.session_state["_resume_parsed_summary"] = parsed
        if parsed.get("branch_guess"):
            st.session_state["branch_select"] = parsed["branch_guess"]
        if parsed.get("cgpa") is not None:
            st.session_state["cgpa_input"] = parsed["cgpa"]
        if parsed.get("projects_count") is not None:
            st.session_state["projects_input"] = parsed["projects_count"]
        if parsed.get("internships_count") is not None:
            st.session_state["internships_input"] = parsed["internships_count"]
        if parsed.get("certifications_count") is not None:
            st.session_state["certifications_input"] = parsed["certifications_count"]
        guessed_role = guess_best_role(parsed.get("known_skills", []))
        if guessed_role:
            st.session_state["role_select"] = guessed_role
        # Applied once branch/role (and therefore the valid skill options) are
        # known below — stash for now, filter to valid options, then apply.
        st.session_state["_resume_pending_skills"] = parsed.get("known_skills", [])
        st.session_state["_resume_applied_key"] = resume_applied_key

    if st.session_state.get("_resume_parsed_summary"):
        parsed_summary = st.session_state["_resume_parsed_summary"]
        with st.expander("📋 What we found in your resume — review before predicting", expanded=True):
            if not parsed_summary.get("text_extracted"):
                st.warning(
                    "We couldn't read any text from this file — it may be a scanned/image-only resume. "
                    "Please fill the form in manually below."
                )
            else:
                found_bits = []
                if parsed_summary.get("branch_guess"):
                    found_bits.append(f"Branch: **{parsed_summary['branch_guess']}**")
                if parsed_summary.get("cgpa") is not None:
                    found_bits.append(f"CGPA: **{parsed_summary['cgpa']}**")
                if parsed_summary.get("projects_count") is not None:
                    found_bits.append(f"Projects: **{parsed_summary['projects_count']}**")
                if parsed_summary.get("internships_count") is not None:
                    found_bits.append(f"Internships: **{parsed_summary['internships_count']}**")
                if parsed_summary.get("certifications_count") is not None:
                    found_bits.append(f"Certifications: **{parsed_summary['certifications_count']}**")
                if found_bits:
                    st.markdown(" · ".join(found_bits))
                else:
                    st.caption("We found the resume text, but couldn't confidently detect CGPA/section counts — "
                               "please fill those in manually.")
                if parsed_summary.get("known_skills"):
                    st.caption("Skills detected (auto-selected below — remove any that don't apply):")
                    st.markdown(
                        " ".join(f'<span class="skill-pill">{s}</span>' for s in parsed_summary["known_skills"]),
                        unsafe_allow_html=True,
                    )
                st.caption("Everything below is editable — this is best-effort extraction, not guaranteed accurate.")

    st.markdown("---")

    dcol1, dcol2 = st.columns(2)
    with dcol1:
        branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "EEE", "MECH", "CIVIL"], key="branch_select")
    with dcol2:
        role_options = ordered_roles_for_branch(branch)
        role = st.selectbox(
            "Job Role you're targeting",
            role_options, key="role_select",
            help="Recommendations (missing skills, readiness score) will be tailored to this role, "
                 "on top of your branch benchmark. The list is ordered for your branch, but you can "
                 "pick any role.",
        )
    skill_catalog = get_combined_catalog(branch, role)

    # Now that branch/role (and so the valid skill options) are resolved,
    # apply any skills detected from a just-uploaded resume.
    pending_skills = st.session_state.pop("_resume_pending_skills", None)
    if pending_skills is not None:
        st.session_state["known_skills_select"] = [s for s in pending_skills if s in skill_catalog]

    with st.form("student_form"):
        col1, col2 = st.columns(2)

        with col1:
            cgpa = st.number_input("CGPA (out of 10)", 0.0, 10.0, step=0.1, key="cgpa_input")
            aptitude = st.slider("Aptitude Score (0-100)", 0, 100, key="aptitude_input")
            communication = st.slider("Communication Skills (0-100)", 0, 100, key="communication_input")

        with col2:
            projects = st.number_input("Number of Projects", 0, 15, key="projects_input")
            internships = st.number_input("Number of Internships", 0, 10, key="internships_input")
            certifications = st.number_input("Number of Certifications", 0, 15, key="certifications_input")
            backlogs = st.number_input("Number of Active Backlogs", 0, 10, key="backlogs_input")

        st.markdown(f"**Technical Skills — select what you know ({branch} + {role})**")
        known_skills = st.multiselect(
            "Skills", options=list(skill_catalog.keys()), label_visibility="collapsed",
            key="known_skills_select",
        )
        technical = compute_technical_score(branch, known_skills)
        st.caption(f"Computed technical skill score based on your selection: **{technical}/100**")

        with st.expander("🔬 Advanced: subject-wise breakdown (optional)"):
            st.caption(
                "Fine-tune your scores per subject for a more precise skill-gap report. "
                "Defaults to your overall scores above if left unchanged."
            )
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                dsa_score = st.slider("DSA", 0, 100, int(technical))
                dbms_score = st.slider("DBMS", 0, 100, int(technical))
            with sc2:
                os_cn_score = st.slider("OS & CN", 0, 100, int(technical))
                quant_score = st.slider("Quant Aptitude", 0, 100, int(aptitude))
            with sc3:
                logical_score = st.slider("Logical Reasoning", 0, 100, int(aptitude))
                verbal_score = st.slider("Verbal Ability", 0, 100, int(aptitude))

        submitted = st.form_submit_button("🚀 Predict Placement & Analyze Skill Gap", use_container_width=True)

    if submitted and artifacts_ready:
        student_input = {
            "CGPA": cgpa, "Aptitude_Score": aptitude, "Technical_Skills": technical,
            "Communication_Skills": communication, "Projects_Count": projects,
            "Internships_Count": internships, "Certifications_Count": certifications,
            "Backlogs": backlogs,
        }
        subject_input = {
            "DSA_Score": dsa_score, "DBMS_Score": dbms_score, "OS_CN_Score": os_cn_score,
            "Quant_Score": quant_score, "Logical_Score": logical_score, "Verbal_Score": verbal_score,
        }

        X = build_feature_row(student_input, branch, branch_encoder)
        X_scaled = scaler.transform(X)
        pred, proba, band = predict_with_uncertainty(model, uncertainty_model, scaler, X_scaled)

        st.markdown("---")
        st.subheader("📊 Placement Prediction")

        res_col1, res_col2 = st.columns([1, 1.3])

        with res_col1:
            gauge_color = "#00D2A0" if proba >= 0.5 else "#FF6B6B"
            steps = [{"range": [0, 40], "color": "#2E1A1A"},
                     {"range": [40, 70], "color": "#2E2A1A"},
                     {"range": [70, 100], "color": "#1A2E24"}]
            threshold = None
            if band is not None:
                lo = max(0, (proba - band) * 100)
                hi = min(100, (proba + band) * 100)
                threshold = {"line": {"color": "#F5F5F7", "width": 2}, "thickness": 0.75, "value": hi}
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 40, "color": "#F5F5F7"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#9A9CB0"},
                    "bar": {"color": gauge_color},
                    "bgcolor": "#1A1D29",
                    "borderwidth": 0,
                    "steps": steps,
                    **({"threshold": threshold} if threshold else {}),
                },
                title={"text": "Placement Probability", "font": {"size": 16, "color": "#9A9CB0"}},
            ))
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", font={"color": "#F5F5F7"})
            st.plotly_chart(fig, use_container_width=True)
            if band is not None:
                st.caption(
                    f"Confidence band: roughly **{max(0,(proba-band)*100):.0f}%–{min(100,(proba+band)*100):.0f}%** "
                    "(spread across an ensemble of decision trees — a wider band means the model is less certain)."
                )

        with res_col2:
            pred_label = "Likely to be PLACED" if pred == 1 else "Currently at risk of not being placed"
            if pred == 1:
                st.success(f"**{pred_label}** — estimated probability: {proba*100:.1f}%")
            else:
                st.error(f"**{pred_label}** — estimated probability: {proba*100:.1f}%")

            mcol1, mcol2, mcol3 = st.columns(3)
            for col, label, val in zip(
                [mcol1, mcol2, mcol3],
                ["CGPA", "Technical Score", "Communication"],
                [f"{cgpa}", f"{technical}", f"{communication}"],
            ):
                col.markdown(f'<div class="metric-card"><div class="value">{val}</div>'
                             f'<div class="label">{label}</div></div>', unsafe_allow_html=True)

            if known_skills:
                st.markdown("**Skills you selected:**")
                st.markdown(" ".join(f'<span class="skill-pill">{s}</span>' for s in known_skills),
                            unsafe_allow_html=True)

        # ---- Peer percentile comparison ----
        st.markdown("---")
        st.subheader("👥 How You Compare to Your Branch")
        branch_df = load_dataset_stats()
        branch_df = branch_df[branch_df["Branch"] == branch]
        from scipy.stats import percentileofscore
        percentile_rows = []
        for feat, val in student_input.items():
            if feat == "Backlogs" or feat not in branch_df.columns:
                continue
            pct = percentileofscore(branch_df[feat], val)
            percentile_rows.append({"Feature": feat.replace("_", " "), "Percentile": round(pct, 1)})
        pct_df = pd.DataFrame(percentile_rows).sort_values("Percentile")
        pct_fig = go.Figure(go.Bar(
            x=pct_df["Percentile"], y=pct_df["Feature"], orientation="h",
            marker_color="#6C5CE7",
        ))
        pct_fig.update_layout(
            height=280, xaxis=dict(range=[0, 100], title="Percentile within your branch"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#F5F5F7"},
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(pct_fig, use_container_width=True)
        st.caption("E.g. a 70th percentile in CGPA means you score higher than ~70% of students in your branch.")

        # ---- Skill gap analysis ----
        st.markdown("---")
        st.subheader("🧩 Skill Gap Analysis")
        full_student = {**student_input, **subject_input}
        result = analyze_skill_gap(full_student, branch=branch, known_skills=known_skills, role=role)
        gap_df = result["gap_table"]

        st.dataframe(
            gap_df.rename(columns={
                "Your_Value": "Your Value", "Benchmark_Value": "Placed-Student Benchmark",
                "Gap_Percent": "Gap (%)", "Priority_Score": "Priority",
            }),
            use_container_width=True, hide_index=True,
        )

        if "technical_subject_gap" in result or "aptitude_subject_gap" in result:
            with st.expander("🔬 Subject-wise breakdown"):
                sub_c1, sub_c2 = st.columns(2)
                if "technical_subject_gap" in result and not result["technical_subject_gap"].empty:
                    with sub_c1:
                        st.markdown("**Technical subjects**")
                        st.dataframe(result["technical_subject_gap"], use_container_width=True, hide_index=True)
                if "aptitude_subject_gap" in result and not result["aptitude_subject_gap"].empty:
                    with sub_c2:
                        st.markdown("**Aptitude subjects**")
                        st.dataframe(result["aptitude_subject_gap"], use_container_width=True, hide_index=True)

        st.subheader("💡 Personalized Recommendations")
        all_recs = list(result["recommendations"]) + list(result.get("subject_recommendations", []))
        if all_recs:
            for rec in all_recs:
                st.markdown(f'<div class="rec-card"><b>{rec["Area"].replace("_", " ")}</b> — {rec["Advice"]}</div>',
                            unsafe_allow_html=True)
        else:
            st.info("Great job! You're at or above the benchmark on all tracked skills.")

        if result.get("missing_skills"):
            st.subheader("🎯 Skills to Learn Next")
            st.caption(f"Highest-impact skills for {branch} students that you haven't listed yet:")
            pills = []
            for m in result["missing_skills"]:
                link = get_resource_link(m["Skill"])
                if link:
                    pills.append(f'<span class="skill-pill"><a href="{link}" target="_blank">{m["Skill"]} 🔗</a></span>')
                else:
                    pills.append(f'<span class="skill-pill">{m["Skill"]}</span>')
            st.markdown(" ".join(pills), unsafe_allow_html=True)

        # ---- Role fit ----
        if result.get("role"):
            st.markdown("---")
            st.subheader(f"🧭 Role Fit — {role}")
            readiness = result.get("role_readiness", 0.0)
            rc1, rc2 = st.columns([1, 2])
            with rc1:
                st.markdown(
                    f'<div class="metric-card"><div class="value">{readiness}%</div>'
                    f'<div class="label">Skill match for {role}</div></div>',
                    unsafe_allow_html=True,
                )
            with rc2:
                if readiness >= 75:
                    st.success(f"Strong match — your selected skills already cover most of what a "
                               f"{role} role typically needs.")
                elif readiness >= 40:
                    st.warning(f"Partial match — you cover some of the core {role} skills, but a "
                               "few high-impact gaps remain below.")
                else:
                    st.error(f"Early stage — most of the skills recruiters expect for {role} aren't "
                             "in your list yet. Use the gaps below to prioritize.")

            if result.get("missing_role_skills"):
                st.caption(f"Highest-impact skills for a **{role}** that you haven't listed yet:")
                role_pills = []
                for m in result["missing_role_skills"]:
                    link = get_role_resource_link(m["Skill"])
                    if link:
                        role_pills.append(
                            f'<span class="skill-pill"><a href="{link}" target="_blank">{m["Skill"]} 🔗</a></span>'
                        )
                    else:
                        role_pills.append(f'<span class="skill-pill">{m["Skill"]}</span>')
                st.markdown(" ".join(role_pills), unsafe_allow_html=True)
            else:
                st.info(f"You already know every catalog skill listed for {role}. 🎉")

        if backlogs > 0:
            st.warning(f"You currently have {backlogs} active backlog(s). Clearing these should be a top "
                       "priority, as backlogs are commonly used as a hard eligibility filter by recruiters.")

        # ---- Explainability (SHAP) ----
        st.markdown("---")
        st.subheader("🔍 Why this prediction? (Explainability)")
        if raw_model is not None:
            try:
                background = scaler.transform(load_dataset_stats()[FEATURE_ORDER[:-1] + ["Branch"]].assign(
                    Branch=branch_encoder.transform(load_dataset_stats()["Branch"])
                ).sample(min(100, len(load_dataset_stats())), random_state=1))
                explainer = get_shap_explainer(raw_model, background)
                shap_values = explainer.shap_values(X_scaled) if hasattr(explainer, "shap_values") else explainer(X_scaled).values
                if isinstance(shap_values, list):
                    sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                else:
                    sv = np.array(shap_values)[0]
                    if sv.ndim > 1:
                        sv = sv[:, -1]
                contrib_df = pd.DataFrame({
                    "Feature": FEATURE_ORDER,
                    "Contribution": sv,
                }).sort_values("Contribution")
                contrib_fig = go.Figure(go.Bar(
                    x=contrib_df["Contribution"], y=contrib_df["Feature"], orientation="h",
                    marker_color=["#FF6B6B" if v < 0 else "#00D2A0" for v in contrib_df["Contribution"]],
                ))
                contrib_fig.update_layout(
                    height=320, title="Feature contribution to your prediction (SHAP values)",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#F5F5F7"},
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(contrib_fig, use_container_width=True)
                st.caption("Green bars pushed your prediction toward 'placed'; red bars pushed it toward 'not placed'.")
            except Exception as e:
                st.info(f"Explainability chart unavailable for this model right now ({e}).")
        else:
            st.info("Raw (uncalibrated) model not found — retrain with the current train_models.py to enable this.")

        # ---- Company eligibility ----
        st.markdown("---")
        st.subheader("🏢 Illustrative Company Eligibility")
        st.caption(
            "Based on editable example criteria in `config/company_criteria.json` — "
            "**not** live recruiter data. Replace with your placement cell's real criteria."
        )
        eligibility = check_eligibility(student_input, branch)
        for c in eligibility:
            css_class = "company-eligible" if c["eligible"] else "company-not-eligible"
            detail = "Meets current example criteria." if c["eligible"] else "; ".join(c["reasons"])
            st.markdown(
                f'<div class="company-card {css_class}"><b>{c["company"]}</b><br>'
                f'<span style="font-size:0.85rem;color:#9A9CB0">{detail}</span></div>',
                unsafe_allow_html=True,
            )

        # ---- Save state for other tabs (What-If, Progress, PDF) ----
        st.session_state["last_result"] = {
            "student_input": student_input,
            "branch": branch,
            "role": role,
            "pred_label": pred_label,
            "proba": proba,
            "gap_rows": gap_df.to_dict("records"),
            "recommendations": all_recs,
            "missing_skills": result.get("missing_skills"),
            "role_readiness": result.get("role_readiness"),
            "missing_role_skills": result.get("missing_role_skills"),
            "eligibility": eligibility,
        }

        # ---- Save to progress tracker ----
        st.markdown("---")
        if student_id.strip():
            if st.button("💾 Save this assessment to My Progress"):
                save_snapshot(student_id, branch, proba, student_input)
                st.success(f"Saved! Check the 'My Progress' tab to see your trend for '{student_id}'.")
        else:
            st.info("Enter a Student ID in the sidebar to save this assessment and track your progress over time.")

        # ---- PDF export ----
        pdf_path = build_report_pdf(
            student_id=student_id, branch=branch, prediction_label=pred_label,
            probability=proba, inputs=student_input, gap_rows=gap_df.to_dict("records"),
            recommendations=all_recs, missing_skills=result.get("missing_skills"),
            eligible_companies=eligibility,
            role=result.get("role"), role_readiness=result.get("role_readiness"),
            missing_role_skills=result.get("missing_role_skills"),
        )
        with open(pdf_path, "rb") as f:
            st.download_button(
                "📄 Download PDF Report", data=f.read(),
                file_name=f"placement_report_{student_id or 'student'}.pdf",
                mime="application/pdf", use_container_width=True,
            )

    st.markdown("---")
    st.caption("Note: Predictions are based on a trained ML model using historical/simulated student data "
               "and are meant to guide preparation, not to be a certain outcome.")

# ============================================================================
# TAB 2 — WHAT-IF SIMULATOR
# ============================================================================
with tab_whatif:
    st.subheader("🔧 What-If Simulator")
    st.caption(
        "See how your placement probability would change if you improved one factor at a time, "
        "holding everything else constant. Run a prediction in the Predict tab first."
    )
    last = st.session_state.get("last_result")
    if not artifacts_ready:
        st.info("Model artifacts not found yet.")
    elif last is None:
        st.info("Submit the form in the 🎯 Predict tab first — this simulator starts from your last assessment.")
    else:
        base_input = last["student_input"]
        branch_wi = last["branch"]
        st.write(f"Baseline: **{base_input}** (Branch: {branch_wi}) → "
                 f"**{last['proba']*100:.1f}%** placement probability")

        wi_feature = st.selectbox(
            "Which factor do you want to explore?",
            ["Internships_Count", "Certifications_Count", "Projects_Count",
             "CGPA", "Technical_Skills", "Communication_Skills", "Aptitude_Score", "Backlogs"],
            format_func=lambda x: x.replace("_", " "),
        )

        ranges = {
            "Internships_Count": np.arange(0, 6),
            "Certifications_Count": np.arange(0, 9),
            "Projects_Count": np.arange(0, 10),
            "CGPA": np.round(np.arange(5.0, 10.1, 0.5), 1),
            "Technical_Skills": np.arange(10, 101, 10),
            "Communication_Skills": np.arange(10, 101, 10),
            "Aptitude_Score": np.arange(10, 101, 10),
            "Backlogs": np.arange(0, 6),
        }
        vals = ranges[wi_feature]
        probs = []
        for v in vals:
            trial = base_input.copy()
            trial[wi_feature] = v
            X_trial = build_feature_row(trial, branch_wi, branch_encoder)
            X_trial_scaled = scaler.transform(X_trial)
            probs.append(model.predict_proba(X_trial_scaled)[0][1] * 100)

        wi_fig = go.Figure(go.Scatter(x=vals, y=probs, mode="lines+markers", line=dict(color="#6C5CE7", width=3)))
        wi_fig.add_vline(x=base_input[wi_feature], line_dash="dash", line_color="#9A9CB0",
                          annotation_text="Your current value")
        wi_fig.update_layout(
            height=380, xaxis_title=wi_feature.replace("_", " "), yaxis_title="Placement Probability (%)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#F5F5F7"},
            yaxis=dict(range=[0, 100], gridcolor="#2E3140"),
        )
        st.plotly_chart(wi_fig, use_container_width=True)

        best_idx = int(np.argmax(probs))
        st.info(
            f"If your **{wi_feature.replace('_', ' ')}** were **{vals[best_idx]}** instead of "
            f"**{base_input[wi_feature]}**, your estimated probability would move from "
            f"**{last['proba']*100:.1f}%** to **{probs[best_idx]:.1f}%** (all else unchanged)."
        )

# ============================================================================
# TAB 3 — BATCH PREDICTION
# ============================================================================
with tab_batch:
    st.subheader("📁 Batch Prediction")
    st.caption(
        "Upload a CSV with columns: CGPA, Aptitude_Score, Technical_Skills, Communication_Skills, "
        "Projects_Count, Internships_Count, Certifications_Count, Backlogs, Branch — get predictions "
        "for a whole class/cohort at once."
    )
    template_df = pd.DataFrame([{
        "CGPA": 7.2, "Aptitude_Score": 60, "Technical_Skills": 55, "Communication_Skills": 58,
        "Projects_Count": 2, "Internships_Count": 1, "Certifications_Count": 1, "Backlogs": 0, "Branch": "CSE",
    }])
    st.download_button(
        "⬇️ Download CSV template", data=template_df.to_csv(index=False),
        file_name="batch_template.csv", mime="text/csv",
    )

    uploaded = st.file_uploader("Upload student CSV", type=["csv"])
    if uploaded is not None and artifacts_ready:
        try:
            batch_df = pd.read_csv(uploaded)
            missing_cols = [c for c in FEATURE_ORDER if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                work = batch_df.copy()
                work["Branch_Encoded"] = branch_encoder.transform(work["Branch"])
                X_batch = work[["CGPA", "Aptitude_Score", "Technical_Skills", "Communication_Skills",
                                 "Projects_Count", "Internships_Count", "Certifications_Count",
                                 "Backlogs"]].copy()
                X_batch["Branch"] = work["Branch_Encoded"]
                X_batch = X_batch[FEATURE_ORDER]
                X_batch_scaled = scaler.transform(X_batch)

                batch_df["Placement_Probability_%"] = (model.predict_proba(X_batch_scaled)[:, 1] * 100).round(1)
                batch_df["Predicted_Status"] = np.where(
                    model.predict(X_batch_scaled) == 1, "Likely Placed", "At Risk"
                )
                batch_df = batch_df.sort_values("Placement_Probability_%")

                st.success(f"Scored {len(batch_df)} students.")
                st.dataframe(batch_df, use_container_width=True, hide_index=True)

                at_risk = batch_df[batch_df["Predicted_Status"] == "At Risk"]
                st.markdown(f"### ⚠️ {len(at_risk)} student(s) currently at risk")
                if len(at_risk):
                    st.dataframe(at_risk, use_container_width=True, hide_index=True)

                st.download_button(
                    "⬇️ Download results CSV", data=batch_df.to_csv(index=False),
                    file_name="batch_predictions.csv", mime="text/csv", use_container_width=True,
                )
        except Exception as e:
            st.error(f"Couldn't process this file: {e}")

# ============================================================================
# TAB 4 — MY PROGRESS
# ============================================================================
with tab_progress:
    st.subheader("📈 My Progress Over Time")
    sid = st.session_state.get("student_id", "").strip()
    if not sid:
        st.info("Enter a Student ID in the sidebar, save an assessment from the 🎯 Predict tab, "
                 "then come back here to see your trend.")
    else:
        history = load_history(sid)
        if not history:
            st.info(f"No saved assessments yet for '{sid}'. Save one from the 🎯 Predict tab.")
        else:
            hist_df = pd.DataFrame(history)
            hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
            fig = go.Figure(go.Scatter(
                x=hist_df["timestamp"], y=hist_df["probability"] * 100,
                mode="lines+markers", line=dict(color="#00D2A0", width=3),
            ))
            fig.update_layout(
                height=350, yaxis_title="Placement Probability (%)", xaxis_title="Date",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#F5F5F7"},
                yaxis=dict(range=[0, 100], gridcolor="#2E3140"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

            if len(hist_df) >= 2:
                delta = (hist_df["probability"].iloc[-1] - hist_df["probability"].iloc[0]) * 100
                verb = "improved" if delta >= 0 else "dropped"
                st.metric("Change since first assessment", f"{delta:+.1f} pts",
                          delta=f"{delta:+.1f}")

            if st.button("🗑️ Clear my saved history"):
                clear_history(sid)
                st.rerun()

# ============================================================================
# TAB 5 — MODEL INSIGHTS
# ============================================================================
with tab_insights:
    st.subheader("📊 Model Performance & Data Insights")

    comparison_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    if os.path.exists(comparison_path):
        comp_df = pd.read_csv(comparison_path)
        st.markdown("#### Model Comparison (tuned, held-out test set)")
        show_cols = [c for c in comp_df.columns if c != "Best_Params"]
        st.dataframe(comp_df[show_cols], use_container_width=True, hide_index=True)
        with st.expander("Best hyperparameters found per model"):
            for _, row in comp_df.iterrows():
                if "Best_Params" in comp_df.columns:
                    st.write(f"**{row['Model']}**: {row['Best_Params']}")

        fig = go.Figure()
        for metric in ["Accuracy", "Precision", "Recall", "F1_Score"]:
            fig.add_trace(go.Bar(name=metric, x=comp_df["Model"], y=comp_df[metric]))
        fig.update_layout(
            barmode="group", height=420, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font={"color": "#F5F5F7"},
            yaxis=dict(range=[0, 1], gridcolor="#2E3140"), legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `python src/train_models.py` to generate model comparison results.")

    cv_path = os.path.join(REPORTS_DIR, "cv_comparison.csv")
    if os.path.exists(cv_path):
        st.markdown("#### 5-Fold Cross-Validation (F1-score, on training data)")
        cv_df = pd.read_csv(cv_path)
        st.dataframe(cv_df, use_container_width=True, hide_index=True)
        st.caption("Mean ± std across 5 folds — a more reliable estimate than a single train/test split.")

    info = load_model_info()
    if info:
        st.markdown("#### Class Balance & Calibration")
        cb = info.get("class_balance", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Placed (train+test)", cb.get("placed", "—"))
        c2.metric("Not placed", cb.get("not_placed", "—"))
        c3.metric("Imbalance ratio", cb.get("imbalance_ratio", "—"))
        if info.get("calibrated_metrics"):
            st.caption(
                f"After probability calibration, the deployed model scores "
                f"F1={info['calibrated_metrics']['F1_Score']}, "
                f"Accuracy={info['calibrated_metrics']['Accuracy']} on the held-out test set."
            )
        st.caption(f"Last trained: {info.get('trained_at', 'unknown')}")

    calib_fig_path = os.path.join(FIGURES_DIR, "08_calibration_curve.png")
    if os.path.exists(calib_fig_path):
        st.image(calib_fig_path, caption="Calibration Curve — closer to the diagonal = more trustworthy probabilities.")

    st.markdown("#### Exploratory Data Analysis")
    figure_files = [
        ("01_class_balance.png", "Placement Status Distribution"),
        ("02_correlation_heatmap.png", "Feature Correlation Heatmap"),
        ("03_cgpa_vs_placement.png", "CGPA vs Placement Status"),
        ("05_placement_rate_by_branch.png", "Placement Rate by Branch"),
        ("07_feature_importance.png", "Feature Importance (Best Model)"),
    ]
    cols = st.columns(2)
    for i, (fname, caption) in enumerate(figure_files):
        fpath = os.path.join(FIGURES_DIR, fname)
        if os.path.exists(fpath):
            with cols[i % 2]:
                st.image(fpath, caption=caption, use_container_width=True)

    if not any(os.path.exists(os.path.join(FIGURES_DIR, f)) for f, _ in figure_files):
        st.info("Run `python src/eda.py` and `python src/train_models.py` to generate charts here.")

# ============================================================================
# TAB 6 — ABOUT
# ============================================================================
with tab_about:
    st.subheader("ℹ️ About This Project")
    st.markdown("""
**Placement Prediction and Skill Gap Analysis Using Machine Learning**

This project predicts whether a student is likely to be placed based on
academic performance, aptitude, technical & communication skills, projects,
internships, certifications, and backlogs — then identifies exactly which
skills to prioritize next.

**Pipeline:**
1. **Data Preprocessing** — cleaning, encoding, scaling
2. **EDA** — correlation analysis, class balance, branch-wise trends
3. **Modeling** — Logistic Regression, Decision Tree, Random Forest, XGBoost,
   each cross-validated and hyperparameter-tuned; class imbalance handled via
   class weighting / `scale_pos_weight`
4. **Calibration** — the winning model is wrapped in `CalibratedClassifierCV`
   so the displayed percentages are trustworthy probabilities, with a
   confidence band from a Random Forest tree-vote ensemble
5. **Evaluation** — Accuracy, Precision, Recall, F1-score, Confusion Matrix,
   calibration curve
6. **Skill Gap Analysis** — benchmarking against successfully placed peers,
   weighted by feature importance, with specific named-skill recommendations
   and subject-wise (DSA/DBMS/OS-CN, Quant/Logical/Verbal) breakdowns
7. **Explainability** — SHAP values show exactly why a given student got
   their prediction
8. **Deployment** — this interactive Streamlit dashboard, with batch
   scoring, a what-if simulator, peer percentile comparison, illustrative
   company-eligibility checks, PDF export, and per-student progress tracking

**Tech Stack:** Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit, Plotly, fpdf2, SQLite

**Note:** The dataset used here is a realistically simulated dataset (not
scraped real student records), built to reflect plausible statistical
relationships between skills and placement outcomes. Swap in a real,
institution-specific dataset (same column names) for production use. The
company-eligibility criteria are illustrative placeholders — edit
`config/company_criteria.json` with real placement-cell data before treating
that section as anything other than a demo.
""")
