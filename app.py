import streamlit as st
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Screener", layout="centered")
st.title("🤖 AI Resume Screener (Local Hugging Face)")

with st.expander("📄 About this App", expanded=True):
    st.write(
        "Compares resume with job description, shows match score, skills, and AI feedback."
    )

resume_input = st.text_area("✍️ Paste Your Resume", height=250)
jd_input = st.text_area("🧾 Paste Job Description", height=250)

# ---------- FIX 1: Load model safely ----------
@st.cache_resource
def load_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

try:
    summarizer = load_model()
except:
    summarizer = None

# ---------- FIX 2: Skill-based scoring ----------
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

# ---------- MAIN BUTTON ----------
if st.button("🔍 Analyze Match"):
    if not resume_input or not jd_input:
        st.warning("Please provide both resume and job description.")
    else:
        # ---------- Cosine similarity ----------
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_input, jd_input])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

        # ---------- Skill score ----------
        score, matched, missing = skill_match(jd_input, resume_input)

        st.markdown(f"### ✅ Similarity Score: `{round(similarity * 100, 2)}%`")
        st.markdown(f"### 🎯 Skill Match Score: `{score}%`")

        st.write("**Matched Skills:**", matched)
        st.write("**Missing Skills:**", missing)

        # ---------- Recommendation ----------
        if score > 75:
            rec = "Shortlist"
        elif score > 50:
            rec = "Consider"
        else:
            rec = "Reject"

        st.write("**Recommendation:**", rec)

        # ---------- FIX 3: Safe summarization ----------
        with st.spinner("Generating AI feedback..."):
            if summarizer:
                try:
                    text = resume_input[:1000] + "\n" + jd_input[:1000]
                    summary = summarizer(text, max_length=80, min_length=20, do_sample=False)
                    st.markdown("### 🧠 AI Feedback Summary:")
                    st.success(summary[0]['summary_text'])
                except Exception as e:
                    st.error(f"Summarization failed: {e}")
            else:
                st.warning("AI feedback unavailable (model failed to load)")