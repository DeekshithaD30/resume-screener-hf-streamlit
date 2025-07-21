# Streamlit Resume Screener (Hugging Face)

An interactive web app that compares a resume with a job description using cosine similarity and Hugging Face summarization — all local, no API keys.

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🧠 Features
- Match score (TF-IDF cosine similarity)
- AI-generated feedback using `distilbart-cnn-12-6` from Hugging Face
