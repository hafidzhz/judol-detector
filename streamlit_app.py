# streamlit_app.py
import streamlit as st
import joblib
import pandas as pd
from detect_gambling_comments import clean_comment_text, normalize_unicode

model = joblib.load('gambling_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

st.title("Comment Gambling Detector")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['cleaned'] = df['comment'].apply(lambda x: clean_comment_text(normalize_unicode(x)))
    X = vectorizer.transform(df['cleaned'])
    df['gambling_prob'] = model.predict_proba(X)[:, 1]
    df['gambling'] = df['gambling_prob'] > 0.5
    st.write(df[['comment', 'gambling_prob', 'gambling']])
