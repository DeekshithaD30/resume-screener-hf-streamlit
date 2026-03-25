import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Resume Screener", layout="centered")
st.title("🤖 AI Resume Screener")

with st.expander("📄 About this App", expanded=True):
    st.write(
        "This tool compares a resume with a job description using skill matching "
        "and text similarity, and provides actionable feedback."
    )

# ---------- INPUT ----------
resume_input = st.text_area("✍️ Paste Your Resume", height=250)
jd_input = st.text_area("🧾 Paste Job Description", height=250)

# ---------- SKILLS ----------
SKILLS = [
    "python", "sql", "tableau", "power bi", "excel",
    "machine learning", "pandas", "numpy", "matplotlib"
]

def extract_skills(text):
    text = text.lower()
    return [skill for skill in SKILLS if skill in text]

def skill_match(jd, resume):
    jd_skills = set(extract_skills(jd))
    resume_skills = set(extract_skills(resume))

    matched = jd_skills & resume_skills
    missing = jd_skills - resume_skills

    if len(jd_skills) == 0:
        return 0, [], []

    score = (len(matched) / len(jd_skills)) * 100
    return round(score, 2), list(matched), list(missing)

# ---------- MAIN ----------
if st.button("🔍 Analyze Match"):
    if not resume_input or not jd_input:
        st.warning("Please provide both resume and job description.")
    else:
        # ---------- TEXT SIMILARITY ----------
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_input, jd_input])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

        # ---------- SKILL MATCH ----------
        score, matched, missing = skill_match(jd_input, resume_input)

        st.markdown(f"### ✅ Text Similarity (TF-IDF): `{round(similarity * 100, 2)}%`")
        st.markdown(f"### 🎯 Skill Match Score: `{score}%`")

        st.markdown("**Matched Skills:** " + (", ".join(matched) if matched else "None"))
        st.markdown("**Missing Skills:** " + (", ".join(missing) if missing else "None"))

        # ---------- RECOMMENDATION ----------
        if score > 75:
            rec = "Shortlist"
        elif score > 50:
            rec = "Consider"
        else:
            rec = "Reject"

        st.markdown(f"**Recommendation:** {rec}")

        # ---------- LIGHTWEIGHT FEEDBACK ----------
        st.markdown("### 🧠 AI Feedback:")

        feedback = []

        if score > 75:
            feedback.append("Strong match for the role.")
        elif score > 50:
            feedback.append("Moderate match. Improvement needed.")
        else:
            feedback.append("Low match. Significant skill gaps.")

        if missing:
            feedback.append("Focus on improving: " + ", ".join(missing))

        if matched:
            feedback.append("Strengths: " + ", ".join(matched))

        for f in feedback:
            st.write("- " + f)