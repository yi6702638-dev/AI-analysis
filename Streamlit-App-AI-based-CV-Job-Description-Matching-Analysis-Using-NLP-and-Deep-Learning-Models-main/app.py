import streamlit as st
from inference import InferenceEngine

st.set_page_config(page_title="CV ↔ JD Matcher", layout="wide")

@st.cache_resource
def load_engine():
    return InferenceEngine(artifacts_dir="artifacts")

engine = load_engine()

st.title("AI-based CV–Job Description Matching Analysis")
st.header("(Using NLP and Deep Learning Models)")

col1, col2 = st.columns(2)
with col1:
    cv_text = st.text_area("CV / Resume Text（Left）", height=350, placeholder="Paste your CV text here…")
with col2:
    jd_text = st.text_area("Job Description（Right）", height=350, placeholder="Paste the job description (JD) text here…")

run = st.button("Results Generated", type="primary", use_container_width=True)

if run:
    if not cv_text.strip() or not jd_text.strip():
        st.warning("Please enter both the CV and the job description (JD).")
    else:
        out = engine.run_all(cv_text, jd_text)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Match Probability (Model)", f"{out['match']['match_prob']:.3f}")
        m2.metric("TF-IDF Cosine", f"{out['similarity']['tfidf_cosine']:.3f}")
        m3.metric("Resume Recommendation (Binary)", out["shortlist"]["decision"], f"{out['shortlist']['shortlist_prob']:.3f}")
        m4.metric("Predicted Job Category (Multi-class)", out["category"]["top_category"], f"{out['category']['top_prob']:.3f}")

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top 10 Category Probabilities")
            st.json(out["category"]["probs"])
        with c2:
            st.subheader("Skill Extraction (CV / JD / Overlap)")
            st.write("**CV skills**:", out["skills"]["cv_skills"])
            st.write("**JD skills**:", out["skills"]["jd_skills"])
            st.write("**Overlap**:", out["skills"]["overlap"])

        with st.expander("View Full JSON Output"):
            st.json(out)

import datetime

year = datetime.datetime.now().year

st.markdown(
    f"""
    <hr style="margin-top: 2rem; margin-bottom: 0.75rem;">
    <div style="text-align:center; color: #888; font-size: 0.9rem;">
        © {year} Qingyang Xiao. All rights reserved.<br>
       The outputs of this tool are model-based scores and similarity references only and do not constitute hiring or employment recommendations.
    </div>
    """,
    unsafe_allow_html=True
)