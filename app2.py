import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
model = load_model('rnn_sentiment_model.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Set parameters
max_len = 100
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Streamlit UI
st.set_page_config(page_title="Coffee Review Sentiment", layout="centered")
st.title("☕ Coffee Review Sentiment Analyzer")
st.write("Enter a review to predict the sentiment:")

user_input = st.text_area("✍️ Your Review", height=150)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a review.")
    else:
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=max_len)
        prediction = model.predict(padded)
        predicted_label = np.argmax(prediction)

        st.subheader("🔍 Sentiment Prediction:")
        st.success(f"**{label_map[predicted_label]}**")

        st.subheader("📊 Probabilities:")
        st.write(f"- Negative: {prediction[0][0]:.2f}")
        st.write(f"- Neutral: {prediction[0][1]:.2f}")
        st.write(f"- Positive: {prediction[0][2]:.2f}")
