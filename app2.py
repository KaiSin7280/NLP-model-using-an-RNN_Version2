import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the updated RNN model saved as .keras
model = load_model('rnn_sentiment_model.keras')  # ✅ CHANGED from .pkl to .keras

# Settings
max_len = 150  # Should match what was used during training

# Prediction function
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded)
    label = np.argmax(prediction, axis=1)[0]
    sentiment = ["Negative", "Neutral", "Positive"][label]
    confidence = np.max(prediction)
    return sentiment, confidence

# Streamlit App
st.set_page_config(page_title="Sentiment Analysis (RNN)", layout="centered")
st.title("💬 Sentiment Classifier (RNN)")

user_input = st.text_area("Enter a review:")
if st.button("Analyze"):
    if user_input.strip():
        sentiment, confidence = predict_sentiment(user_input)
        st.markdown(f"### 🧠 Sentiment: **{sentiment}**")
        st.markdown(f"### 🔍 Confidence: **{confidence:.2f}**")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
