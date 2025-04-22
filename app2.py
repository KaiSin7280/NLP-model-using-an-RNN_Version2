import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained RNN model and tokenizer
model = joblib.load('rnn_sentiment_model.pkl')
tokenizer = joblib.load('tokenizer.pkl')

# Constants
max_len = 100
labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

# App UI
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("☕ Coffee Review Sentiment Analyzer (RNN)")
st.markdown("Enter a review to predict its sentiment:")

user_input = st.text_area("Review", "")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        # Preprocess input
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=max_len)

        # Predict
        prediction = model.predict(padded)
        predicted_class = np.argmax(prediction)

        # Display result
        st.subheader("Prediction:")
        st.write(f"**{labels[predicted_class]}**")
        st.progress(float(np.max(prediction)))

        # Show class probabilities
        st.subheader("Confidence:")
        for idx, prob in enumerate(prediction[0]):
            st.write(f"{labels[idx]}: {prob:.2f}")
