import streamlit as st
import pickle
import re
import string
import numpy as np
from nltk.corpus import stopwords
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import MaxAbsScaler  # Import MaxAbsScaler

# Load models
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
with open('xgboost_model.pkl', 'rb') as f:
    xgboost_model = pickle.load(f)

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

# Define word lists
profane_words = ["shit", "fuck", "ass", "bitch", "cunt", "dick", "piss", "cock", "bastard", "faggot", "idiot", "stupid", "moron", "trash", "garbage", 'HELL', "hell"]
toxicity_terms = {
    "insult": ["idiot", "stupid", "moron", "fool", "jerk", "loser"],
    "hate_speech": ["racist", "sexist", "bigot", "nazi"],
    "threat": ["kill", "hit", "hurt", "destroy", "attack"],
    "harassment": ["stalk", "bully", "abuse", "mock", "tease"],
    "offensive": ["trash", "garbage", "worthless", "pathetic"]
}

# Prediction Function
def predict_toxicity(comment_text):
    cleaned_text = clean_text(comment_text)

    # Create features for individual profane words
    profane_features = [cleaned_text.count(word) for word in profane_words]

    # Create features for individual toxicity terms
    toxicity_features = []
    for category, terms in toxicity_terms.items():
        for word in terms:
            toxicity_features.append(cleaned_text.count(word))

    # TF-IDF Vectorization
    tfidf_matrix = tfidf_vectorizer.transform([cleaned_text])

    # Combine Features
    X_additional = np.array(profane_features + toxicity_features).reshape(1, -1)

    # Scale Additional Features
    scaler = MaxAbsScaler()
    X_additional_scaled = scaler.fit_transform(X_additional)
    X_additional_sparse = csr_matrix(X_additional_scaled)

    X = hstack([tfidf_matrix, X_additional_sparse])

    # Make Prediction
    prediction = xgboost_model.predict(X)[0]
    return prediction

# Streamlit App
st.title("Toxic Comment Classifier")

comment_text = st.text_area("Enter a comment to analyze:", "")

if st.button("Predict"):
    prediction = predict_toxicity(comment_text)
    if prediction == 1:
        st.error("⚠️ This comment is classified as **toxic**.  Consider revising your words.")
    else:
        st.success("✅ This comment is classified as **non-toxic**.  Keep the conversation healthy!")