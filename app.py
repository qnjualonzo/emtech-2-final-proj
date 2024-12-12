import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('sentiment_analysis_model.h5')  # Replace with your model path

# Define the parameters
max_features = 10000
maxlen = 500

# Streamlit app title
st.title("Sentiment Analysis Web App")

# Input text for prediction
st.subheader("Enter a movie review:")
user_input = st.text_area("Type your review here:")

# Define a function to process input and predict sentiment
def predict_sentiment(text):
    # Load the IMDb tokenizer to preprocess the text (use the same preprocessing steps)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts([text])  # Fit the tokenizer to the user input
    sequence = tokenizer.texts_to_sequences([text])  # Convert text to sequence
    padded_sequence = pad_sequences(sequence, maxlen=maxlen)  # Pad the sequence

    # Make a prediction using the loaded model
    prediction = model.predict(padded_sequence)
    return prediction

# Button to trigger prediction
if st.button("Predict Sentiment"):
    if user_input:
        # Make prediction
        prediction = predict_sentiment(user_input)

        # Display result based on prediction
        if prediction >= 0.5:
            st.success("The sentiment of the review is **Positive**!")
        else:
            st.error("The sentiment of the review is **Negative**!")
    else:
        st.warning("Please enter a review to analyze!")

