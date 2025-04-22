import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load model
model = load_model('rnn_sentiment_model.keras')
max_len = 150

# Prediction function
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded)
    label = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))
    sentiment = ["Negative", "Neutral", "Positive"][label]
    emoji = ["😠", "😐", "😊"][label]
    return sentiment, emoji, confidence

# Batch prediction function
def predict_batch(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len)
    predictions = model.predict(padded)
    results = []
    for i, pred in enumerate(predictions):
        label = np.argmax(pred)
        confidence = float(np.max(pred))
        sentiment = ["Negative", "Neutral", "Positive"][label]
        emoji = ["😠", "😐", "😊"][label]
        results.append({
            "Review": texts[i],
            "Sentiment": sentiment,
            "Emoji": emoji,
            "Confidence": round(confidence, 2)
        })
    return pd.DataFrame(results)

# Streamlit App
st.set_page_config(page_title="RNN Sentiment Classifier", layout="centered")
st.title("💬 Sentiment Classifier (RNN)")
st.write("Analyze single or multiple reviews for sentiment (Positive, Neutral, Negative).")

tab1, tab2 = st.tabs(["🗣️ Single Review", "📁 CSV Batch Upload"])

# Single Review Tab
with tab1:
    st.subheader("🧍 Enter a review:")
    user_input = st.text_area("Type a review here", height=150)

    if st.button("🔍 Analyze Review"):
        if user_input.strip():
            sentiment, emoji, confidence = predict_sentiment(user_input)
            st.markdown(f"""
                ### {emoji} Sentiment: **{sentiment}**  
                #### 🔎 Confidence: **{confidence:.2f}**
            """)
        else:
            st.warning("⚠️ Please enter a review.")

# CSV Upload Tab
with tab2:
    st.subheader("📤 Upload a CSV file with a 'reviews' column:")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if 'reviews' not in df.columns:
                st.error("❌ CSV must have a column named 'reviews'")
            else:
                st.success(f"✅ {len(df)} reviews loaded. Click below to analyze.")
                if st.button("🧠 Analyze All Reviews"):
                    results_df = predict_batch(df['reviews'].tolist())
                    st.markdown("### ✅ Batch Results:")
                    st.dataframe(results_df)

                    # Download
                    csv_data = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_data,
                        file_name="sentiment_results.csv",
                        mime='text/csv'
                    )
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
