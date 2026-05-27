import streamlit as st
import pickle
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
# -----------------------------
# Load Saved Files
# -----------------------------
model = load_model("mental_health_model.h5")

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

stop_words = set(stopwords.words('english'))

MAX_LENGTH = 50

# -----------------------------
# Text Preprocessing
# -----------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# -----------------------------
# Prediction Function
# -----------------------------
def predict_emotion(text):
    cleaned = preprocess_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post')

    prediction = model.predict(padded)

    predicted_index = np.argmax(prediction)

    confidence = np.max(prediction)

    sentiment = label_encoder.inverse_transform([predicted_index])[0]

    return sentiment, confidence, prediction[0]

# -----------------------------
# Emotional Guidance
# -----------------------------
def get_guidance(sentiment):
    guidance = {
        "Anxiety": "Take a short break and practice deep breathing.",
        "Depression": "Talk to someone you trust and focus on small positive steps.",
        "Stress": "Try relaxation exercises or a short walk.",
        "Normal": "Keep maintaining your positive emotional balance."
    }

    return guidance.get(sentiment, "Stay mindful and prioritize self-care.")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Mental Health Sentiment Monitor", layout="wide")

# SECTION 1 — Header
st.title("AI-Based Mental Health Sentiment Monitoring System")
st.subheader("Emotion Detection using Simple Recurrent Neural Networks")

# SECTION 2 — About Project
st.markdown("## About the Project")
st.write("""
This project uses Natural Language Processing and Simple RNN models
to analyze emotional sentiment from user text.

Importance:
- Early emotional pattern detection
- Mental wellness monitoring
- AI-assisted emotional analysis

RNN Role:
Simple RNN learns sequence patterns and remembers previous words
to understand emotional context.
""")

# SECTION 3 — User Input
st.markdown("## Enter Your Thoughts")

user_text = st.text_area(
    "Input Text",
    placeholder="Enter your thoughts or feelings here..."
)

st.write("Sample Inputs:")
st.info("I feel anxious about tomorrow")
st.info("I am feeling calm and peaceful today")
st.info("Everything feels overwhelming lately")

# SECTION 4 — Prediction Button
if st.button("Analyze Emotion"):

    if user_text.strip() == "":
        st.warning("Please enter text.")

    else:
        sentiment, confidence, probs = predict_emotion(user_text)

        # SECTION 5 — Prediction Output
        st.markdown("## Prediction Result")

        st.success(f"Emotion Detected: {sentiment}")
        st.info(f"Confidence: {confidence*100:.2f}%")

        emotional_status = "Positive" if sentiment == "Normal" else "Needs Attention"
        st.warning(f"Emotional Status: {emotional_status}")

        # SECTION 6 — Visualization
        st.markdown("## Sentiment Confidence Visualization")

        labels = label_encoder.classes_

        fig, ax = plt.subplots()
        ax.bar(labels, probs)
        ax.set_ylabel("Probability")
        ax.set_title("Sentiment Confidence")

        st.pyplot(fig)

        # SECTION 7 — Emotional Guidance
        st.markdown("## Emotional Guidance")

        guidance = get_guidance(sentiment)

        st.write(guidance)
