import streamlit as st
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Screener", layout="centered")
st.title("🤖 AI Resume Screener (Local Hugging Face)")

with st.expander("📄 About this App", expanded=True):
    st.write(
        "This tool compares your resume to a job description using cosine similarity, "
        "and gives AI-generated feedback using a Hugging Face summarization model."
    )

resume_input = st.text_area("✍️ Paste Your Resume", height=250)
jd_input = st.text_area("🧾 Paste Job Description", height=250)

if st.button("🔍 Analyze Match"):
    if not resume_input or not jd_input:
        st.warning("Please provide both resume and job description.")
    else:
        # Cosine similarity
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_input, jd_input])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        st.markdown(f"### ✅ Resume Match Score: `{round(similarity * 100, 2)}%`")

        # Hugging Face summarizer
        with st.spinner("Generating AI feedback..."):
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            prompt = f"Compare this resume to the job description and provide suggestions.\nResume: {resume_input}\nJD: {jd_input}"
            summary = summarizer(prompt, max_length=130, min_length=30, do_sample=False)
            st.markdown("### 🧠 AI Feedback Summary:")
            st.success(summary[0]['summary_text'])
