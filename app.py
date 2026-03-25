import streamlit as st
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Resume Screener", layout="centered")
st.title("🤖 AI Resume Screener")

with st.expander("📄 About this App", expanded=True):
    st.write(
        "This tool compares a resume with a job description using skill matching, "
        "text similarity, and AI-generated feedback."
    )

# ---------- INPUT ----------
resume_input = st.text_area("✍️ Paste Your Resume", height=250)
jd_input = st.text_area("🧾 Paste Job Description", height=250)

# ---------- LOAD MODEL SAFELY ----------
@st.cache_resource
def load_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

try:
    summarizer = load_model()
except:
    summarizer = None

# ---------- SKILL LIST ----------
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

# ---------- MAIN ACTION ----------
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

        st.markdown("**Matched Skills:** " + ", ".join(matched) if matched else "None")
        st.markdown("**Missing Skills:** " + ", ".join(missing) if missing else "None")

        # ---------- RECOMMENDATION ----------
        if score > 75:
            rec = "Shortlist"
        elif score > 50:
            rec = "Consider"
        else:
            rec = "Reject"

        st.markdown(f"**Recommendation:** {rec}")

        # ---------- AI FEEDBACK ----------
        with st.spinner("Generating AI feedback..."):
            if summarizer:
                try:
                    prompt = f"""
                    Compare the following resume with the job description.

                    Provide:
                    - Key strengths
                    - Missing skills
                    - Suggestions for improvement

                    Resume:
                    {resume_input[:800]}

                    Job Description:
                    {jd_input[:800]}
                    """

                    summary = summarizer(prompt, max_length=100, min_length=30, do_sample=False)

                    st.markdown("### 🧠 AI Feedback Summary:")
                    st.success(summary[0]['summary_text'])

                except Exception as e:
                    st.error(f"AI feedback failed: {e}")
            else:
                st.warning("AI feedback unavailable (model failed to load)")