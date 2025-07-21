# 🤖 Resume Screener — Hugging Face + Streamlit

An AI-powered resume screening web app built using Hugging Face Transformers and Streamlit. Compares a resume to a job description using cosine similarity, and gives AI-generated feedback — fully local, no OpenAI key required.

---

## 🚀 Features

- ✅ Cosine Similarity Resume-JD Match Scoring  
- 🧠 AI-Powered Feedback using `distilbart-cnn-12-6`
- 💻 Streamlit UI — paste your resume + JD and get real-time insights
- 🔐 100% local — no OpenAI API, no billing, no quota limits

---

## 🛠️ Tech Stack

- Python · Streamlit · scikit-learn  
- Transformers (`sshleifer/distilbart-cnn-12-6`)  
- Hugging Face 🤗

---

## 🧪 How to Run Locally

```bash
git clone https://github.com/DeekshithaD30/resume-screener-hf-streamlit.git
cd resume-screener-hf-streamlit
pip install -r requirements.txt
streamlit run app.py
