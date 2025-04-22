import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the updated RNN model saved as .keras
model = load_model('rnn_sentiment_model.keras')

# Settings
max_len = 150  # Should match training settings

# Prediction function
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded)
    label = np.argmax(prediction, axis=1)[0]
    sentiment = ["Negative", "Neutral", "Positive"][label]
    confidence = np.max(prediction)
    return sentiment, confidence

# Streamlit UI Configuration
st.set_page_config(page_title="Sentiment Classifier (RNN)", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f6f9fc;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.6em 1.2em;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1 style='text-align: center;'>💬 Sentiment Classifier (RNN)</h1>", unsafe_allow_html=True)
st.write(" ")

# Text Input
user_input = st.text_area("✍️ Enter a customer review below:", height=150)

# Analyze Button
if st.button("🚀 Analyze Sentiment"):
    if user_input.strip():
        sentiment, confidence = predict_sentiment(user_input)

        color_map = {
            "Positive": "green",
            "Neutral": "orange",
            "Negative": "red"
        }

        st.markdown(f"""
            <h3 style='text-align: center;'>🧠 Prediction</h3>
            <div style='text-align: center; font-size: 24px; color: {color_map[sentiment]};'>
                Sentiment: <strong>{sentiment}</strong>
            </div>
            <div style='text-align: center; font-size: 20px; margin-top: 10px;'>
                Confidence: <strong>{confidence:.2f}</strong>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text to analyze.")
