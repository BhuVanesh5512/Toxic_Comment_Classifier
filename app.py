import streamlit as st
import pickle
import re
import string
import numpy as np
nltk.download("stopwords")
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
    "hate_speech": ["racist", "sexist", "bigot", "nazi","rapist"],
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
st.set_page_config(page_title="Toxic Comment Classifier", layout="wide")

# Custom CSS for better readability
st.markdown("""
    <style>
        body { background-color: #f0f2f6; }
        .stApp { background-color: #f8f9fc; }
        h1 { color: #003366; font-weight: bold; }
        h2, h3 { color: #004488; }
        p, label { color: #222222; font-size: 18px; }
        .stTextArea, .stButton { font-size: 18px; }
        .stTextArea textarea { background-color: #ffffff; color: #222222; border-radius: 10px; }
        .stButton button { background-color: #0073e6; color: white; font-weight: bold; padding: 10px; border-radius: 8px; }
        .stButton button:hover { background-color: #005bb5; }
        .sidebar .sidebar-content { background-color: #e6f2ff; padding: 20px; border-radius: 10px; }
        @media (prefers-color-scheme: dark) {
            body, .stApp { background-color: #121212; color: white; }
            h1, h2, h3 { color: #ffffff; }
            p, label { color: #dddddd; }
            .stTextArea textarea { background-color: #333333; color: white; }
            .stButton button { background-color: #005bb5; color: white; }
            .sidebar .sidebar-content { background-color: #222222; }
        }
    </style>
""", unsafe_allow_html=True)

# Title to model
st.markdown("""
    <h1 style="color: #003366; font-weight: bold;">Toxic Comment Classifier</h1>
    <h3 style="color: #004488;">Understanding Toxicity in Online Discussions</h3>
    <p style="color: #004386; font-size: 18px;">
        Toxic comments can negatively impact online conversations, making them hostile or unwelcoming. 
        Our machine learning model helps detect such comments to promote healthier discussions.
    </p>
""", unsafe_allow_html=True)

# Input Section
comment_text = st.text_area("Enter a comment to analyze:", "")

if st.button("Predict"):
    prediction = predict_toxicity(comment_text)
    if prediction == 1:
        st.error("⚠️ This comment is classified as **toxic**.  Consider revising your words.")
    else:
        st.success("✅ This comment is classified as **non-toxic**.  Keep the conversation healthy!")

# Sidebar Information
st.sidebar.header("About Toxic Comments")
st.sidebar.markdown("""
Toxic comments include:
- **Insults & Personal Attacks**
- **Hate Speech**
- **Threats of Violence**
- **Harassment & Bullying**
- **Offensive Language**
""")

st.sidebar.header("Why Detect Toxicity?")
st.sidebar.markdown("""
- 🌍 Promote inclusive online spaces
- 🛡️ Protect users from online abuse
- 💬 Encourage constructive conversations
""")

st.sidebar.info("Please use this tool responsibly.This page only classifies toxic comment. Predictions may not always be accurate.")
