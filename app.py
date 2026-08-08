# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load trained artifacts once, when the app starts
model = joblib.load('models/logreg_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

# Exact SAME preprocessing functions as the notebook
def extract_numeric_features(text):
    num_exclamations = text.count('!')
    has_currency_symbol = 1 if re.search(r'[$£€]', text) else 0
    pct_uppercase = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    num_digits = sum(1 for c in text if c.isdigit())
    return [num_exclamations, has_currency_symbol, pct_uppercase, num_digits]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)

# Streamlit UI
st.title("📩 SMS Spam Detector")
st.write("Classify a text message as spam or ham using TF-IDF + Logistic Regression.")

user_input = st.text_area("Enter a message:", height=100)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        numeric_feats = extract_numeric_features(user_input)
        cleaned = clean_text(user_input)
        final_text = preprocess_text(cleaned)

        text_vector = tfidf.transform([final_text])

        from scipy.sparse import hstack
        import numpy as np
        combined = hstack([text_vector, [numeric_feats]])

        prediction = model.predict(combined)[0]
        probability = model.predict_proba(combined)[0][1]

        if prediction == 1:
            st.error(f"🚨 SPAM (confidence: {probability:.1%})")
        else:
            st.success(f"✅ HAM (confidence: {1 - probability:.1%})")
