import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer and model
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = load_model('rnn_sentiment_model.keras')  # ✅ Changed from .pkl to .keras

# Settings
max_len = 150  # Should match training setting

# Prediction function (single)
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded)
    label = np.argmax(prediction, axis=1)[0]
    sentiment_list = ["Negative", "Neutral", "Positive"]
    emoji_list = ["😠", "😐", "😊"]
    return sentiment_list[label], emoji_list[label], np.max(prediction)

# Prediction function (batch)
def predict_batch(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len)
    predictions = model.predict(padded)
    labels = np.argmax(predictions, axis=1)
    sentiments = [ ["Negative", "Neutral", "Positive"][i] for i in labels ]
    emojis = [ ["😠", "😐", "😊"][i] for i in labels ]
    confidences = np.max(predictions, axis=1)
    return pd.DataFrame({
        "Review": texts,
        "Sentiment": sentiments,
        "Emoji": emojis,
        "Confidence": confidences
    })

# Streamlit layout
st.set_page_config(page_title="Sentiment Analyzer (RNN)", layout="centered")
st.title("💬 Sentiment Classifier using RNN")
st.markdown("Predict if a review is **Positive**, **Neutral**, or **Negative** using a trained RNN model.")

tab1, tab2 = st.tabs(["🧠 Single Prediction", "📁 Batch Analysis"])

# Tab 1: Single Prediction
with tab1:
    user_input = st.text_area("Enter a review:")
    if st.button("🔍 Analyze"):
        if user_input.strip():
            sentiment, emoji, confidence = predict_sentiment(user_input)
            st.markdown(f"### Sentiment: **{sentiment}** {emoji}")
            st.markdown(f"### Confidence: `{confidence:.2f}`")
        else:
            st.warning("⚠️ Please enter a review to analyze.")

# Tab 2: Batch Prediction from CSV
with tab2:
    st.markdown("Upload a CSV file containing a column named `reviews`.")
    uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "reviews" not in df.columns:
            st.error("❌ The CSV must have a column named 'reviews'.")
        else:
            if st.button("🧠 Analyze All Reviews"):
                results_df = predict_batch(df['reviews'].tolist())

                # Display results
                st.markdown("### ✅ Batch Results:")
                st.dataframe(results_df)

                # Bar chart of sentiment counts
                st.markdown("### 📊 Sentiment Distribution")
                sentiment_counts = results_df['Sentiment'].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
                st.bar_chart(sentiment_counts)

                # Download results
                csv_data = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name="sentiment_results.csv",
                    mime='text/csv'
                )
